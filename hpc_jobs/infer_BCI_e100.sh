#!/bin/bash
#SBATCH --job-name=asp_infer_BCI_e100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/asp/logs/infer_BCI_e100.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/asp/logs/infer_BCI_e100.%j.err

# infer_BCI_e100.sh
# Runs inference on the full BCI test split using the latest checkpoint
# from the BCI 100-epoch training run.
#
# Inference uses --preprocess none --load_size 1024 --crop_size 1024 so that
# images pass through at full 1024x1024 resolution (no random crop).
# --phase val: BCI-AB.sqsh has valA/valB (not testA/testB). Output goes to val_latest/.
# --no_flip ensures deterministic ordering.
#
# Submit ONLY after submit_BCI_e100.sh has completed successfully.
# Submit: sbatch infer_BCI_e100.sh
#
# Output images land at:
#   $VSC_DATA/projects/asp/outputs/results/BCI_e100/val_latest/images/fake_B/
#
# Verify after job:
#   find $VSC_DATA/projects/asp/outputs/results/BCI_e100 -name "*.png" | wc -l
#   Expected: 977

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/asp_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/asp/code/asp"
CHECKPOINTS_DIR="$VSC_DATA/projects/asp/outputs/checkpoints"
RESULTS_DIR="$VSC_DATA/projects/asp/outputs/results"
RUN_NAME="BCI_e100"
BCI_ASP_SQSH="$VSC_SCRATCH/BCI-AB.sqsh"
BCI_ASP_MNT="$VSC_SCRATCH/sqsh_mnt/BCI-AB"

# =========================
# MODULES
# =========================

module purge
module load calcua/2026.1

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Container ==="
echo "  $CONTAINER"
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

echo ""
echo "=== Environment ==="
apptainer exec --nv "$CONTAINER" python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo ""
echo "=== Checkpoint check ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint folder not found: $CKPT_DIR"
    echo "Has submit_BCI_e100.sh completed successfully?"
    exit 1
fi
echo "  Checkpoints found:"
find "$CKPT_DIR" -name "*.pth" | sort

echo ""
echo "=== Test dataset check ==="
mkdir -p "$BCI_ASP_MNT"
apptainer exec \
    -B "$BCI_ASP_SQSH:$BCI_ASP_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  testA: \$(ls $BCI_ASP_MNT/testA | wc -l) images\"; echo \"  testB: \$(ls $BCI_ASP_MNT/testB | wc -l) images\""

mkdir -p "$RESULTS_DIR/$RUN_NAME"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/asp/logs/gpu_infer_BCI_e100.csv" & GPU_LOG_PID=$!

# =========================
# INFERENCE
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting BCI inference ==="
echo "  run name    : $RUN_NAME"
echo "  results dir : $RESULTS_DIR/$RUN_NAME"
echo "  dataroot    : $BCI_ASP_MNT (inside BCI-asp.sqsh)"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$BCI_ASP_SQSH:$BCI_ASP_MNT:image-src=/" \
    "$CONTAINER" \
    python test.py \
        --dataroot        "$BCI_ASP_MNT" \
        --name            "$RUN_NAME" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --results_dir     "$RESULTS_DIR" \
        --model           cpt \
        --CUT_mode        FastCUT \
        --dataset_mode    aligned \
        --direction       AtoB \
        --netG            resnet_6blocks \
        --normG           instance \
        --weight_norm     spectral \
        --no_dropout \
        --nce_layers      0,4,8,12,16 \
        --load_size       1024 \
        --crop_size       1024 \
        --preprocess      none \
        --phase           val \
        --num_test        9999 \
        --no_flip \
        --display_id      -1 \
        --gpu_ids         0

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID

echo ""
echo "=== Output image count ==="
find "$RESULTS_DIR/$RUN_NAME" -name "*.png" | wc -l

echo ""
echo "=== Output folder structure ==="
ls "$RESULTS_DIR/$RUN_NAME/val_latest/images/" 2>/dev/null || echo "WARNING: val_latest/images/ not found"

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/asp/logs/gpu_infer_BCI_e100.csv"

echo ""
echo "BCI inference complete. Next step: sbatch eval_BCI_e100.sh"
