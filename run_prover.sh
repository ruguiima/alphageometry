#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-alphageometry}"
MELIAD_PATH="${MELIAD_PATH:-$ROOT_DIR/meliad_lib}"
DATA="${DATA:-/root/autodl-tmp/alphageometry/ag_ckpt_vocab}"

MODE="${1:-${MODE:-alphageometry}}"
if [[ "$MODE" == "ddar" && $# -lt 2 && -z "${PROBLEM_NAME:-}" ]]; then
  PROBLEM_NAME="translated_imo_2000_p1"
  PROBLEMS_FILE="${PROBLEMS_FILE:-$ROOT_DIR/imo_ag_30.txt}"
else
  PROBLEM_NAME="${2:-${PROBLEM_NAME:-orthocenter}}"
  PROBLEMS_FILE="${3:-${PROBLEMS_FILE:-$ROOT_DIR/examples.txt}}"
fi

BATCH_SIZE="${BATCH_SIZE:-2}"
BEAM_SIZE="${BEAM_SIZE:-2}"
DEPTH="${DEPTH:-2}"

if [[ "$MODE" != "alphageometry" && "$MODE" != "ddar" ]]; then
  echo "MODE must be either 'alphageometry' or 'ddar'." >&2
  echo "Usage: $0 [alphageometry|ddar] [problem_name] [problems_file]" >&2
  exit 2
fi

for path in "$ROOT_DIR/defs.txt" "$ROOT_DIR/rules.txt" "$PROBLEMS_FILE"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if [[ ! -d "$MELIAD_PATH/transformer/configs" ]]; then
  echo "Missing meliad checkout at: $MELIAD_PATH" >&2
  exit 1
fi

DDAR_ARGS=(
  --defs_file="$ROOT_DIR/defs.txt"
  --rules_file="$ROOT_DIR/rules.txt"
)

SEARCH_ARGS=(
  --beam_size="$BEAM_SIZE"
  --search_depth="$DEPTH"
)

LM_ARGS=(
  --ckpt_path="$DATA"
  --vocab_path="$DATA/geometry.757.model"
  --gin_search_paths="$MELIAD_PATH/transformer/configs,$ROOT_DIR"
  --gin_file=base_htrans.gin
  --gin_file=size/medium_150M.gin
  --gin_file=options/positions_t5.gin
  --gin_file=options/lr_cosine_decay.gin
  --gin_file=options/seq_1024_nocache.gin
  --gin_file=geometry_150M_generate.gin
  --gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True
  --gin_param=TransformerTaskConfig.batch_size="$BATCH_SIZE"
  --gin_param=TransformerTaskConfig.sequence_length=128
  --gin_param=Trainer.restore_state_variables=False
)

if [[ "$MODE" == "alphageometry" ]]; then
  for path in "$DATA/checkpoint_10999999" "$DATA/geometry.757.model"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing model file: $path" >&2
      exit 1
    fi
  done
fi

echo "AlphaGeometry launcher"
echo "  mode:          $MODE"
echo "  problem:       $PROBLEM_NAME"
echo "  problems file: $PROBLEMS_FILE"
echo "  conda env:     $CONDA_ENV"
echo "  meliad:        $MELIAD_PATH"
if [[ "$MODE" == "alphageometry" ]]; then
  echo "  data:          $DATA"
  echo "  batch/beam/depth: $BATCH_SIZE/$BEAM_SIZE/$DEPTH"
fi
echo

CMD=(
  python -m alphageometry
  --alsologtostderr
  --problems_file="$PROBLEMS_FILE"
  --problem_name="$PROBLEM_NAME"
  --mode="$MODE"
  "${DDAR_ARGS[@]}"
)

if [[ "$MODE" == "alphageometry" ]]; then
  CMD+=("${SEARCH_ARGS[@]}" "${LM_ARGS[@]}")
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

cd "$ROOT_DIR"
exec conda run --no-capture-output -n "$CONDA_ENV" \
  env PYTHONPATH="$MELIAD_PATH:${PYTHONPATH:-}" \
  "${CMD[@]}"
