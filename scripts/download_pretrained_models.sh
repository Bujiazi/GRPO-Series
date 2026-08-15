#!/usr/bin/env bash
# One-click download of all pretrained checkpoints required by GRPO-Family into pretrained_models/.
#
# Dependency: huggingface_hub (pip install -U huggingface_hub)
# Usage:
#   bash download_pretrained_models.sh            # download everything
#   bash download_pretrained_models.sh FLUX       # download only the given items (space-separated)
#   bash download_pretrained_models.sh --help     # list available items
#
# Available items: FLUX HPSv2 HPSv3 Qwen2-VL
#
# For users behind a firewall, set a mirror:
#   export HF_ENDPOINT=https://hf-mirror.com
# For gated repos (FLUX.1-dev / Qwen2-VL), login first:
#   huggingface-cli login
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRETRAINED_DIR="${PROJECT_DIR}/pretrained_models"

ALL_ITEMS="FLUX HPSv2 HPSv3 Qwen2-VL"

print_help() {
    echo "Usage: bash $0 [items ...]"
    echo ""
    echo "Available items (none => download all):"
    echo "  FLUX       black-forest-labs/FLUX.1-dev             (full dir, requires license acceptance + login)"
    echo "  HPSv2      HPSv2 head + OpenCLIP ViT-H-14 backbone  (xswu/HPSv2 + laion/...)"
    echo "  HPSv3      HPSv3 reward weights                     (MizzenAI/HPSv3)"
    echo "  Qwen2-VL   Qwen2-VL-7B-Instruct                    (full dir, requires login)"
    echo ""
    echo "Environment variables:"
    echo "  HF_ENDPOINT   HuggingFace mirror, e.g. https://hf-mirror.com"
    exit 0
}

SELECTED=()
for arg in "$@"; do
    case "$arg" in
        -h|--help) print_help ;;
        FLUX|HPSv2|HPSv3|Qwen2-VL) SELECTED+=("$arg") ;;
        *) echo "Unknown item: $arg" >&2; print_help ;;
    esac
done
if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=(FLUX HPSv2 HPSv3 Qwen2-VL)
fi

want() {
    local name="$1"
    for s in "${SELECTED[@]}"; do
        [ "$s" == "$name" ] && return 0
    done
    return 1
}

echo "=============================================="
echo " Target dir: ${PRETRAINED_DIR}"
echo " Items:      ${SELECTED[*]}"
if [ -n "${HF_ENDPOINT:-}" ]; then
    echo " Mirror:     ${HF_ENDPOINT}"
fi
echo "=============================================="

if want FLUX; then
    echo ""
    echo "[1/4] Downloading FLUX.1-dev -> pretrained_models/FLUX/"
    echo "      Note: gated repo. Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev and run 'huggingface-cli login' first."
    huggingface-cli download black-forest-labs/FLUX.1-dev \
        --local-dir "${PRETRAINED_DIR}/FLUX" \
        --local-dir-use-symlinks False
fi

if want HPSv2; then
    echo ""
    echo "[2/4] Downloading HPSv2 head -> pretrained_models/HPSv2/HPS_v2.1_compressed.pt"
    huggingface-cli download xswu/HPSv2 \
        HPS_v2.1_compressed.pt \
        --local-dir "${PRETRAINED_DIR}/HPSv2" \
        --local-dir-use-symlinks False

    echo ""
    echo "[2/4] Downloading OpenCLIP ViT-H-14 backbone -> pretrained_models/HPSv2/open_clip_pytorch_model.bin"
    huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
        open_clip_pytorch_model.bin \
        --local-dir "${PRETRAINED_DIR}/HPSv2" \
        --local-dir-use-symlinks False
fi

if want HPSv3; then
    echo ""
    echo "[3/4] Downloading HPSv3 reward weights -> pretrained_models/HPSv3/HPSv3.safetensors"
    huggingface-cli download MizzenAI/HPSv3 \
        HPSv3.safetensors \
        --local-dir "${PRETRAINED_DIR}/HPSv3" \
        --local-dir-use-symlinks False
fi

if want Qwen2-VL; then
    echo ""
    echo "[4/4] Downloading Qwen2-VL-7B-Instruct -> pretrained_models/HPSv3/Qwen2-VL-7B-Instruct/"
    echo "      Note: gated repo. Run 'huggingface-cli login' first."
    huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
        --local-dir "${PRETRAINED_DIR}/HPSv3/Qwen2-VL-7B-Instruct" \
        --local-dir-use-symlinks False
fi

echo ""
echo "=============================================="
echo " Done. Final directory layout:"
echo "=============================================="
find "${PRETRAINED_DIR}" -maxdepth 2 -not -path '*/.*' | sort
