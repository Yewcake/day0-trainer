#!/usr/bin/env bash
set -euo pipefail

# =================================================================================
# Krea 2 Direct Diffusers LoRA Trainer
# Drop this .sh, train_krea2_lora_direct.py, and one dataset .zip into /workspace.
# Then run:
#   bash Train_Krea2_Direct_Diffusers.sh
#
# This is intentionally not AI-Toolkit. It uses Diffusers + PEFT directly.
# =================================================================================

WORKDIR="${WORKDIR:-/workspace}"
RUN_DIR="${WORKDIR}/krea2-direct"
DATA_DIR="${RUN_DIR}/dataset"
OUT_DIR="${RUN_DIR}/output"
FINAL_OUT_DIR="${WORKDIR}/final_loras_krea2"
SCRIPT_PATH="${WORKDIR}/train_krea2_lora_direct.py"

say()  { echo -e "\033[1;32m[krea2-trainer]\033[0m $*"; }
warn() { echo -e "\033[1;33m[krea2-trainer]\033[0m $*"; }
die()  { echo -e "\033[1;31m[krea2-trainer]\033[0m $*" >&2; exit 1; }

cat <<'EOF'
Y   Y EEEEE W   W  CCCC   A   K   K EEEEE
 Y Y  E     W   W C      A A  K  K  E
  Y   EEEE  W W W C     AAAAA KKK   EEEE
  Y   E     WW WW C     A   A K  K  E
  Y   EEEEE W   W  CCCC A   A K   K EEEEE

        Day0 Krea 2 LoRA Trainer
        Made by Yewcake
EOF
mkdir -p "${RUN_DIR}" "${DATA_DIR}" "${OUT_DIR}" "${FINAL_OUT_DIR}"

if [ ! -f "${SCRIPT_PATH}" ]; then
  die "Missing ${SCRIPT_PATH}. Keep train_krea2_lora_direct.py next to this shell script."
fi

read -rp "Enter your Hugging Face token [blank uses existing login/env HF_TOKEN]: " HF_INPUT
export HF_TOKEN="${HF_INPUT:-${HF_TOKEN:-}}"
if [ -n "${HF_TOKEN}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

read -rp "Enter your WandB API key [blank disables W&B; env WANDB_API_KEY is used if set]: " WANDB_INPUT
export WANDB_API_KEY="${WANDB_INPUT:-${WANDB_API_KEY:-}}"
if [ -n "${WANDB_API_KEY}" ]; then
  ENABLE_WANDB="1"
  read -rp "WandB project name [krea2-lora]: " WANDB_PROJECT_INPUT
  WANDB_PROJECT="${WANDB_PROJECT_INPUT:-krea2-lora}"
else
  ENABLE_WANDB="0"
  WANDB_PROJECT="krea2-lora"
fi

read -rp "LoRA output name [${WANDB_PROJECT}]: " LORA_NAME_INPUT
LORA_NAME="${LORA_NAME_INPUT:-${WANDB_PROJECT}}"
LORA_NAME_SAFE="$(echo "${LORA_NAME}" | tr -cs '[:alnum:]_.-' '_' | sed 's/^_*//; s/_*$//')"
[ -n "${LORA_NAME_SAFE}" ] || LORA_NAME_SAFE="krea2_lora"

read -rp "Krea2 model repo/path [krea/Krea-2-Raw; type auto only if you need detection]: " MODEL_INPUT
KREA2_MODEL_INPUT="${MODEL_INPUT:-${KREA2_MODEL:-krea/Krea-2-Raw}}"

read -rp "Enter your trigger word [L1V14M1R3L]: " TRIGGER_INPUT
TRIGGER_WORD="${TRIGGER_INPUT:-L1V14M1R3L}"

NETWORK_TYPE="lora"
LOKR_FACTOR=16
LOKR_FULL_RANK=0
SAMPLE_LORA_SCALE="1.35"
read -rp "Network type [lora/lokr] (default: lora): " NETWORK_INPUT
case "${NETWORK_INPUT:-lora}" in
  lokr|LoKr|LOKR)
    NETWORK_TYPE="lokr"
    SAMPLE_LORA_SCALE="1.0"
    read -rp "LoKr factor [16; higher = smaller file, -1 = auto largest]: " LOKR_FACTOR_INPUT
    LOKR_FACTOR="${LOKR_FACTOR_INPUT:-16}"
    read -rp "LoKr full rank? [0/1; 1 = best quality but very large files]: " LOKR_FR_INPUT
    LOKR_FULL_RANK="${LOKR_FR_INPUT:-0}"
    ;;
  lora|LoRA|LORA) ;;
  *) die "Invalid network type '${NETWORK_INPUT}'. Use lora or lokr." ;;
esac

shopt -s nullglob
zip_files=("${WORKDIR}"/*.zip)
shopt -u nullglob

filtered_zips=()
for z in "${zip_files[@]}"; do
  base="$(basename "$z")"
  case "$base" in
    *trainer*|*Trainer*|WANtrainer*|*loras*|*LoRAs*) ;;
    *) filtered_zips+=("$z") ;;
  esac
done

if [ "${#filtered_zips[@]}" -eq 1 ]; then
  DATASET_ZIP="${filtered_zips[0]}"
  say "Found dataset: ${DATASET_ZIP}"
elif [ "${#filtered_zips[@]}" -gt 1 ]; then
  echo "Multiple dataset zips found:"
  select z in "${filtered_zips[@]}"; do
    [ -n "$z" ] || die "Invalid selection."
    DATASET_ZIP="$z"
    break
  done
else
  die "No dataset .zip found in ${WORKDIR}."
fi


TRAIN_RES=1024
MAX_STEPS=2000
SAVE_EVERY=250
SAVE_EVERY_EPOCHS=0
SAMPLE_EVERY=0
SAMPLE_STEPS=12
TRAIN_BATCH=1
LORA_RANK=64
LORA_ALPHA=64
LEARNING_RATE="8e-5"
WARMUP_STEPS=100
SEED=42
TARGET_MODULES="character"
OPTIMIZER="paged_adamw8bit"
FP8_BASE=0
GRADIENT_CHECKPOINTING=1
TRAIN_DTYPE="bf16"
LORA_DTYPE="match"
SAVE_DTYPE="bf16"
LORA_TYPE="character"
ENABLE_BUCKETS=1
BUCKET_NO_UPSCALE=1
BUCKET_STEP=16
MIN_BUCKET_RES=384
MAX_BUCKET_AREA=0
VAE_TILING=1
VAE_SLICING=1
ATTENTION_BACKEND="auto"
EMPTY_CACHE_EVERY=0
TRANSFORMER_GROUP_OFFLOAD=0
GROUP_OFFLOAD_BLOCKS=1

echo "------------------------------------------------"
echo "LoRA type:"
echo "  [1] Character / identity  - main Krea2 blocks + txtfusion"
echo "  [2] Style / pose          - main Krea2 blocks only"
echo "  [3] Custom target modules - enter exact preset/list later"
read -rp "Choose LoRA type [1/2/3] (default: 1): " LORA_TYPE_CHOICE
case "${LORA_TYPE_CHOICE:-1}" in
  2)
    LORA_TYPE="style"
    TARGET_MODULES="style"
    ;;
  3)
    LORA_TYPE="custom"
    TARGET_MODULES="character"
    ;;
  *)
    LORA_TYPE="character"
    TARGET_MODULES="character"
    ;;
esac


echo "------------------------------------------------"
echo "Training presets:"
echo "  [1] Fast probe     - 768px max bucket, 1000 steps, rank 32, samples off"
echo "  [2] Balanced       - 1024px max bucket, 2000 steps, rank 64, OOM-safe defaults"
echo "  [3] Full quality   - 1024px max bucket, 3000 steps, rank 64, OOM-safe defaults"
read -rp "Choose preset [1/2/3] (default: 2): " PRESET_CHOICE
case "${PRESET_CHOICE:-2}" in
  1)
    TRAIN_RES=768
    MAX_STEPS=1000
    SAVE_EVERY=250
    SAVE_EVERY_EPOCHS=0
    SAMPLE_EVERY=0
    SAMPLE_STEPS=10
    TRAIN_BATCH=1
    LORA_RANK=32
    LORA_ALPHA=32
    LEARNING_RATE="1e-4"
    WARMUP_STEPS=50
    TARGET_MODULES="character"
    OPTIMIZER="paged_adamw8bit"
    ;;
  3)
    TRAIN_RES=1024
    MAX_STEPS=3000
    SAVE_EVERY=250
    SAVE_EVERY_EPOCHS=0
    SAMPLE_EVERY=0
    SAMPLE_STEPS=12
    TRAIN_BATCH=1
    LORA_RANK=64
    LORA_ALPHA=64
    LEARNING_RATE="7e-5"
    WARMUP_STEPS=100
    TARGET_MODULES="character"
    OPTIMIZER="paged_adamw8bit"
    FP8_BASE=0
    GRADIENT_CHECKPOINTING=1
    ;;
  *)
    ;;
esac
# Re-apply LoRA type target preset after training preset selection.
case "${LORA_TYPE}" in
  style|pose)
    TARGET_MODULES="style"
    ;;
  custom)
    :
    ;;
  *)
    TARGET_MODULES="character"
    ;;
esac

echo "Default settings: res=${TRAIN_RES}, steps=${MAX_STEPS}, save_every=${SAVE_EVERY}, save_epochs=${SAVE_EVERY_EPOCHS}, sample_every=${SAMPLE_EVERY}, sample_steps=${SAMPLE_STEPS}, batch=${TRAIN_BATCH}, rank=${LORA_RANK}, lr=${LEARNING_RATE}, optimizer=${OPTIMIZER}, fp8_base=${FP8_BASE}, gradient_checkpointing=${GRADIENT_CHECKPOINTING}, train_dtype=${TRAIN_DTYPE}, lora_dtype=${LORA_DTYPE}, save_dtype=${SAVE_DTYPE}, scheduler=cosine, lora_type=${LORA_TYPE}, buckets=${ENABLE_BUCKETS}, no_upscale=${BUCKET_NO_UPSCALE}, vae_tiling=${VAE_TILING}, attention_backend=${ATTENTION_BACKEND}"
echo "Save guidance: with ~48 images and batch=1, 5 epochs is ~240 steps. Step saves every 250 are cleaner than duplicate epoch saves."
read -rp "Customize training settings? (y/N): " CUSTOM_SETTINGS
if [[ "${CUSTOM_SETTINGS}" =~ ^[Yy]$ ]]; then
  read -rp "Training resolution [${TRAIN_RES}]: " INPUT_RES
  TRAIN_RES="${INPUT_RES:-${TRAIN_RES}}"

  read -rp "Max training steps [${MAX_STEPS}]: " INPUT_STEPS
  MAX_STEPS="${INPUT_STEPS:-${MAX_STEPS}}"

  read -rp "Save every N steps [${SAVE_EVERY}]: " INPUT_SAVE
  SAVE_EVERY="${INPUT_SAVE:-${SAVE_EVERY}}"

  read -rp "Also save every N epochs [${SAVE_EVERY_EPOCHS}; 0 disables epoch saves; 5 is ~240 steps for 48 images]: " INPUT_SAVE_EPOCHS
  SAVE_EVERY_EPOCHS="${INPUT_SAVE_EPOCHS:-${SAVE_EVERY_EPOCHS}}"

  read -rp "Sample every N steps [${SAMPLE_EVERY}; 0 disables samples]: " INPUT_SAMPLE_EVERY
  SAMPLE_EVERY="${INPUT_SAMPLE_EVERY:-${SAMPLE_EVERY}}"

  read -rp "Sample inference steps [${SAMPLE_STEPS}]: " INPUT_SAMPLE_STEPS
  SAMPLE_STEPS="${INPUT_SAMPLE_STEPS:-${SAMPLE_STEPS}}"

  read -rp "Train batch size [${TRAIN_BATCH}; keep 1 for 1024 identity/full-target Krea2]: " INPUT_BATCH
  TRAIN_BATCH="${INPUT_BATCH:-${TRAIN_BATCH}}"

  read -rp "LoRA rank [${LORA_RANK}; use 32 if memory is tight]: " INPUT_RANK
  LORA_RANK="${INPUT_RANK:-${LORA_RANK}}"
  LORA_ALPHA="${LORA_RANK}"

  read -rp "Learning rate [${LEARNING_RATE}]: " INPUT_LR
  LEARNING_RATE="${INPUT_LR:-${LEARNING_RATE}}"

  read -rp "Warmup steps [${WARMUP_STEPS}]: " INPUT_WARMUP
  WARMUP_STEPS="${INPUT_WARMUP:-${WARMUP_STEPS}}"

  read -rp "Target modules [${TARGET_MODULES}; character=blocks+txtfusion, style=blocks only, attention_only=lower VRAM fallback]: " INPUT_TARGETS
  TARGET_MODULES="${INPUT_TARGETS:-${TARGET_MODULES}}"

  read -rp "Optimizer [${OPTIMIZER}; options: paged_adamw8bit, adamw8bit, adamw_fused, adamw]: " INPUT_OPTIMIZER
  OPTIMIZER="${INPUT_OPTIMIZER:-${OPTIMIZER}}"

  read -rp "Try FP8 layerwise base casting? Experimental. [${FP8_BASE}]: " INPUT_FP8
  FP8_BASE="${INPUT_FP8:-${FP8_BASE}}"

  read -rp "Training compute dtype [${TRAIN_DTYPE}; bf16 recommended, fp32 is much slower/heavier]: " INPUT_TRAIN_DTYPE
  TRAIN_DTYPE="${INPUT_TRAIN_DTYPE:-${TRAIN_DTYPE}}"

  read -rp "LoRA parameter dtype [${LORA_DTYPE}; match recommended for 1024/96GB, fp32 can OOM]: " INPUT_LORA_DTYPE
  LORA_DTYPE="${INPUT_LORA_DTYPE:-${LORA_DTYPE}}"

  read -rp "Saved LoRA dtype [${SAVE_DTYPE}; bf16 matches AI-Toolkit/Comfy-style Krea2 LoRAs]: " INPUT_SAVE_DTYPE
  SAVE_DTYPE="${INPUT_SAVE_DTYPE:-${SAVE_DTYPE}}"

  read -rp "Use gradient checkpointing? Slower but saves VRAM. [${GRADIENT_CHECKPOINTING}; recommended 1]: " INPUT_GC
  GRADIENT_CHECKPOINTING="${INPUT_GC:-${GRADIENT_CHECKPOINTING}}"

  read -rp "Enable aspect-ratio buckets? [${ENABLE_BUCKETS}; recommended 1]: " INPUT_BUCKETS
  ENABLE_BUCKETS="${INPUT_BUCKETS:-${ENABLE_BUCKETS}}"

  read -rp "Do not upscale smaller images? [${BUCKET_NO_UPSCALE}; recommended 1]: " INPUT_NO_UPSCALE
  BUCKET_NO_UPSCALE="${INPUT_NO_UPSCALE:-${BUCKET_NO_UPSCALE}}"

  read -rp "Bucket step/divisibility [${BUCKET_STEP}; Krea2-safe 16]: " INPUT_BUCKET_STEP
  BUCKET_STEP="${INPUT_BUCKET_STEP:-${BUCKET_STEP}}"

  read -rp "Minimum bucket resolution [${MIN_BUCKET_RES}; ignored for no-upscale tiny images]: " INPUT_MIN_BUCKET
  MIN_BUCKET_RES="${INPUT_MIN_BUCKET:-${MIN_BUCKET_RES}}"

  read -rp "Max bucket area [${MAX_BUCKET_AREA}; 0 means resolution²]: " INPUT_MAX_AREA
  MAX_BUCKET_AREA="${INPUT_MAX_AREA:-${MAX_BUCKET_AREA}}"

  read -rp "Enable VAE tiling during cache encode? [${VAE_TILING}; recommended 1]: " INPUT_VAE_TILING
  VAE_TILING="${INPUT_VAE_TILING:-${VAE_TILING}}"

  read -rp "Enable VAE slicing during cache encode? [${VAE_SLICING}; recommended 1]: " INPUT_VAE_SLICING
  VAE_SLICING="${INPUT_VAE_SLICING:-${VAE_SLICING}}"

  read -rp "Attention backend [${ATTENTION_BACKEND}; auto/default/flash/sdpa]: " INPUT_ATTN_BACKEND
  ATTENTION_BACKEND="${INPUT_ATTN_BACKEND:-${ATTENTION_BACKEND}}"

  read -rp "Empty CUDA cache every N steps [${EMPTY_CACHE_EVERY}; 0 disables]: " INPUT_EMPTY_CACHE
  EMPTY_CACHE_EVERY="${INPUT_EMPTY_CACHE:-${EMPTY_CACHE_EVERY}}"

  read -rp "Experimental Diffusers transformer group offload? [${TRANSFORMER_GROUP_OFFLOAD}; 0 safest]: " INPUT_GROUP_OFFLOAD
  TRANSFORMER_GROUP_OFFLOAD="${INPUT_GROUP_OFFLOAD:-${TRANSFORMER_GROUP_OFFLOAD}}"

  read -rp "Group offload blocks per group [${GROUP_OFFLOAD_BLOCKS}]: " INPUT_GROUP_BLOCKS
  GROUP_OFFLOAD_BLOCKS="${INPUT_GROUP_BLOCKS:-${GROUP_OFFLOAD_BLOCKS}}"
elif [[ "${CUSTOM_SETTINGS}" =~ ^[0-9]+$ ]]; then
  TRAIN_RES="${CUSTOM_SETTINGS}"
  warn "Interpreted '${CUSTOM_SETTINGS}' as training resolution. Keeping the other default settings."
fi

case "${TRAIN_DTYPE}" in
  bf16|fp32) ;;
  *) die "Invalid training dtype '${TRAIN_DTYPE}'. Use bf16 or fp32." ;;
esac
case "${LORA_DTYPE}" in
  fp32|match) ;;
  *) die "Invalid LoRA dtype '${LORA_DTYPE}'. Use fp32 or match." ;;
esac
if [ "${LORA_DTYPE}" = "fp32" ] && [ "${TRAIN_RES}" -ge 1024 ]; then
  warn "FP32 LoRA params at ${TRAIN_RES}px can promote activations to fp32 and OOM even on 96GB. Use lora_dtype=match unless you are intentionally testing this."
fi
if [ "${TRAIN_BATCH}" -gt 1 ] && [ "${TRAIN_RES}" -ge 1024 ] && [ "${TARGET_MODULES}" = "character" ]; then
  warn "batch=${TRAIN_BATCH} at ${TRAIN_RES}px with identity/full-target Krea2 usually OOMs on 96GB. Use batch=1, or enable gradient checkpointing if batch=1 still OOMs."
fi
if [ "${FP8_BASE}" = "1" ]; then
  die "FP8 base is disabled for this direct PEFT trainer. It casts LoRA layers to Float8 and crashes. Use gradient_checkpointing=1 or target_modules=attention_only instead."
fi
case "${SAVE_DTYPE}" in
  bf16|fp32) ;;
  *) die "Invalid save dtype '${SAVE_DTYPE}'. Use bf16 or fp32." ;;
esac
case "${OPTIMIZER}" in
  paged_adamw8bit|adamw8bit|adamw_fused|adamw) ;;
  *) die "Invalid optimizer '${OPTIMIZER}'. Use paged_adamw8bit, adamw8bit, adamw_fused, or adamw." ;;
esac
case "${LORA_TYPE}" in
  character|style|pose|custom) ;;
  *) die "Invalid LoRA type '${LORA_TYPE}'." ;;
esac

if [ "${NETWORK_TYPE}" = "lokr" ]; then
  if [ "${LOKR_FULL_RANK}" = "1" ]; then
    NETWORK_TAG="lokr_fullrank_f${LOKR_FACTOR}"
  else
    NETWORK_TAG="lokr_r${LORA_RANK}_f${LOKR_FACTOR}"
  fi
else
  NETWORK_TAG="lora_r${LORA_RANK}"
fi
RUN_NAME="${TRIGGER_WORD}_krea2_${NETWORK_TAG}_${MAX_STEPS}steps"
say "Network: ${NETWORK_TYPE} (${NETWORK_TAG}), sample scale: ${SAMPLE_LORA_SCALE}"
say "Settings: model=${KREA2_MODEL_INPUT}, trigger=${TRIGGER_WORD}, res=${TRAIN_RES}, steps=${MAX_STEPS}, save=${SAVE_EVERY}, save_epochs=${SAVE_EVERY_EPOCHS}, sample=${SAMPLE_EVERY}, sample_steps=${SAMPLE_STEPS}, batch=${TRAIN_BATCH}, rank=${LORA_RANK}, lr=${LEARNING_RATE}, optimizer=${OPTIMIZER}, train_dtype=${TRAIN_DTYPE}, lora_dtype=${LORA_DTYPE}, save_dtype=${SAVE_DTYPE}, scheduler=cosine, lora_type=${LORA_TYPE}, buckets=${ENABLE_BUCKETS}, no_upscale=${BUCKET_NO_UPSCALE}, vae_tiling=${VAE_TILING}, attention_backend=${ATTENTION_BACKEND}"

OVERWRITE_RUN=1
if [ -d "${OUT_DIR}/${RUN_NAME}" ]; then
  warn "Existing run folder found: ${OUT_DIR}/${RUN_NAME}"
  read -rp "Overwrite previous Krea2 run/cache before training? [Y/n]: " OVERWRITE_INPUT
  if [[ "${OVERWRITE_INPUT}" =~ ^[Nn]$ ]]; then
    OVERWRITE_RUN=0
    warn "Keeping previous run folder. The Python trainer will still rebuild cache if model/resolution metadata changed."
  else
    say "Removing previous run folder so this run starts clean..."
    rm -rf "${OUT_DIR:?}/${RUN_NAME}"
  fi
fi

cd "${RUN_DIR}"
if [ ! -d "venv" ]; then
  say "Creating virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate

say "Installing dependencies..."
python -m pip install -U pip wheel packaging
say "Installing PyTorch CUDA 12.8 wheels..."
python -m pip install --progress-bar on torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
say "Installing Diffusers/Transformers from GitHub main for fresh Krea2 support..."
if ! python -m pip install --progress-bar on -U git+https://github.com/huggingface/diffusers.git git+https://github.com/huggingface/transformers.git; then
  warn "GitHub install failed. Trying latest/pre-release PyPI Diffusers and Transformers instead."
  python -m pip install --progress-bar on -U --pre diffusers transformers
fi
python -m pip install --progress-bar on -U accelerate peft safetensors pillow tqdm wandb bitsandbytes hf-transfer sentencepiece protobuf

export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export PYTORCH_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"

say "Resolving a usable Krea2 Diffusers model path..."
MODEL_CANDIDATES_FILE="${RUN_DIR}/model_candidates.txt"
: > "${MODEL_CANDIDATES_FILE}"

add_candidate() {
  local candidate="$1"
  [ -n "${candidate}" ] || return 0
  if ! grep -Fxq "${candidate}" "${MODEL_CANDIDATES_FILE}"; then
    echo "${candidate}" >> "${MODEL_CANDIDATES_FILE}"
  fi
}

if [ "${KREA2_MODEL_INPUT}" != "auto" ]; then
  IFS=',' read -ra USER_CANDIDATES <<< "${KREA2_MODEL_INPUT}"
  for candidate in "${USER_CANDIDATES[@]}"; do
    candidate="$(echo "${candidate}" | xargs)"
    add_candidate "${candidate}"
  done
else
  warn "Auto-detect mode enabled. Official/local Raw candidates are tried before any community fallback."
fi

for local_dir in \
  "${WORKDIR}/krea2-diffusers" \
  "${WORKDIR}/Krea-2-Raw" \
  "${WORKDIR}/krea-2-raw" \
  "${WORKDIR}/models/krea2" \
  "${WORKDIR}/models/Krea-2-Raw" \
  "${WORKDIR}/models/krea-2-raw" \
  "${WORKDIR}/Krea-2-Base-Diffusers" \
  "${WORKDIR}/models/Krea-2-Base-Diffusers"; do
  if [ -d "${local_dir}" ]; then
    add_candidate "${local_dir}"
  fi
done

add_candidate "krea/Krea-2-Raw"
if [ "${ALLOW_COMMUNITY_KREA2_FALLBACK:-0}" = "1" ]; then
  warn "Community Krea2 fallback is enabled. Use only if official Raw/local Diffusers folders fail."
  add_candidate "CalamitousFelicitousness/Krea-2-Base-Diffusers"
fi

RESOLVED_MODEL="$(python - "${MODEL_CANDIDATES_FILE}" <<'PY'
import sys
from pathlib import Path

from diffusers import Krea2Pipeline

candidate_file = Path(sys.argv[1])
candidates = [line.strip() for line in candidate_file.read_text().splitlines() if line.strip()]
errors = []

for candidate in candidates:
    try:
        Krea2Pipeline.load_config(candidate)
        print(candidate)
        raise SystemExit(0)
    except Exception as exc:
        errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

print("No usable Krea2 Diffusers model was found.", file=sys.stderr)
print("Tried:", file=sys.stderr)
for err in errors:
    print(f" - {err}", file=sys.stderr)
raise SystemExit(1)
PY
)"

say "Using Krea2 model: ${RESOLVED_MODEL}"
case "${RESOLVED_MODEL}" in
  CalamitousFelicitousness/*)
    warn "Community fallback selected. If you see huge 'weights not used/newly initialized' warnings, stop and use krea/Krea-2-Raw or a local converted folder instead."
    ;;
esac

say "Preparing dataset..."
find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
unzip -q "${DATASET_ZIP}" -d "${DATA_DIR}"

say "Starting direct Krea2 LoRA training..."
python "${SCRIPT_PATH}" \
  --trainer_brand "Day0 Made by Yewcake" \
  --pretrained_model_name_or_path "${RESOLVED_MODEL}" \
  --dataset_dir "${DATA_DIR}" \
  --output_dir "${OUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --trigger_word "${TRIGGER_WORD}" \
  --lora_type "${LORA_TYPE}" \
  --resolution "${TRAIN_RES}" \
  --enable_buckets "${ENABLE_BUCKETS}" \
  --bucket_no_upscale "${BUCKET_NO_UPSCALE}" \
  --bucket_step "${BUCKET_STEP}" \
  --min_bucket_res "${MIN_BUCKET_RES}" \
  --max_bucket_area "${MAX_BUCKET_AREA}" \
  --vae_tiling "${VAE_TILING}" \
  --vae_slicing "${VAE_SLICING}" \
  --train_batch_size "${TRAIN_BATCH}" \
  --max_train_steps "${MAX_STEPS}" \
  --save_every_n_steps "${SAVE_EVERY}" \
  --save_every_n_epochs "${SAVE_EVERY_EPOCHS}" \
  --sample_every_n_steps "${SAMPLE_EVERY}" \
  --sample_num_inference_steps "${SAMPLE_STEPS}" \
  --sample_lora_scale "${SAMPLE_LORA_SCALE}" \
  --network_type "${NETWORK_TYPE}" \
  --lokr_factor "${LOKR_FACTOR}" \
  --lokr_full_rank "${LOKR_FULL_RANK}" \
  --sample_prompts "A realistic smartphone portrait photo of ${TRIGGER_WORD} woman, same person as the training photos, long dark hair, natural expression, fully clothed, daylight, no text, no watermark.||A candid car selfie photo of ${TRIGGER_WORD} woman, same person as the training photos, long dark hair, black top, sunglasses on head, daylight car interior, no text, no watermark.||A casual outdoor sidewalk selfie photo of ${TRIGGER_WORD} woman, same person as the training photos, long dark hair, relaxed expression, softly blurred city background, no text, no watermark." \
  --rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler "cosine" \
  --lr_warmup_steps "${WARMUP_STEPS}" \
  --target_modules "${TARGET_MODULES}" \
  --optimizer "${OPTIMIZER}" \
  --seed "${SEED}" \
  --enable_wandb "${ENABLE_WANDB}" \
  --wandb_project "${WANDB_PROJECT}" \
  --fp8_base "${FP8_BASE}" \
  --train_dtype "${TRAIN_DTYPE}" \
  --lora_dtype "${LORA_DTYPE}" \
  --save_dtype "${SAVE_DTYPE}" \
  --attention_backend "${ATTENTION_BACKEND}" \
  --empty_cache_every_n_steps "${EMPTY_CACHE_EVERY}" \
  --transformer_group_offload "${TRANSFORMER_GROUP_OFFLOAD}" \
  --group_offload_blocks "${GROUP_OFFLOAD_BLOCKS}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"

say "Copying LoRA checkpoints to ${FINAL_OUT_DIR}..."
while IFS= read -r -d '' checkpoint_dir; do
  checkpoint_id="$(basename "${checkpoint_dir}")"
  step_tag="${checkpoint_id#step-}"
  step_tag="step-${step_tag}"

  copy_lora_variant() {
    local src="$1"
    local suffix="$2"
    if [ -f "${src}" ]; then
      cp "${src}" "${FINAL_OUT_DIR}/${LORA_NAME_SAFE}_${step_tag}_${suffix}.safetensors"
    fi
  }

  copy_lora_variant "${checkpoint_dir}/krea2_comfy_native_lora.safetensors" "${LORA_TYPE}"

  for meta_file in training_args.json lora_key_manifest.json; do
    if [ -f "${checkpoint_dir}/${meta_file}" ]; then
      cp "${checkpoint_dir}/${meta_file}" "${FINAL_OUT_DIR}/${LORA_NAME_SAFE}_${step_tag}_${meta_file}"
    fi
  done
done < <(find "${OUT_DIR}/${RUN_NAME}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -print0 2>/dev/null | sort -z)

say "Training complete. Checkpoints are in: ${FINAL_OUT_DIR}"
