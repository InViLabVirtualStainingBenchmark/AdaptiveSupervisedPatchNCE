#!/bin/bash
#SBATCH --job-name=asp_infer_MIST_e100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/asp/logs/infer_MIST_e100.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/asp/logs/infer_MIST_e100.%j.err

# infer_MIST_e100.sh
# Runs inference on the full MIST-HER2 val split using the latest checkpoint
# from the MIST 100-epoch training run.
#
# MIST-HER2.sqsh has valA/valB at its top level. The dataloader falls back
# to valA/valB automatically when --phase test is used and testA is absent.
# Output is written to test_latest/images/fake_B/ as usual.
#
# Submit ONLY after submit_MIST_e100.sh has completed successfully.
# Submit: sbatch infer_MIST_e100.sh
#
# Output images land at:
#   $VSC_DATA/projects/asp/outputs/results/MIST_e100/test_latest/images/fake_B/
#
# Verify after job:
#   find $VSC_DATA/projects/asp/outputs/results/MIST_e100 -name "*.jpg" | wc -l
#   Expected: 1000

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/asp_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/asp/code/asp"
CHECKPOINTS_DIR="$VSC_DATA/projects/asp/outputs/checkpoints"
RESULTS_DIR="$VSC_DATA/projects/asp/outputs/results"
RUN_NAME="MIST_e100"
MIST_SQSH="$VSC_SCRATCH/MIST-HER2.sqsh"
MIST_MNT="$VSC_SCRATCH/sqsh_mnt/MIST-HER2"

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
    echo "Has submit_MIST_e100.sh completed successfully?"
    exit 1
fi
echo "  Checkpoints found:"
find "$CKPT_DIR" -name "*.pth" | sort

echo ""
echo "=== Val dataset check ==="
mkdir -p "$MIST_MNT"
apptainer exec \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  valA: \$(ls $MIST_MNT/valA | wc -l) images\"; echo \"  valB: \$(ls $MIST_MNT/valB | wc -l) images\""

mkdir -p "$RESULTS_DIR/$RUN_NAME"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/asp/logs/gpu_infer_MIST_e100.csv" & GPU_LOG_PID=$!

# =========================
# INFERENCE
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting MIST inference ==="
echo "  run name    : $RUN_NAME"
echo "  results dir : $RESULTS_DIR/$RUN_NAME"
echo "  dataroot    : $MIST_MNT (inside MIST-HER2.sqsh)"

apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    python test.py \
        --dataroot        "$MIST_MNT" \
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
        --phase           test \
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
find "$RESULTS_DIR/$RUN_NAME" -name "*.jpg" | wc -l

echo ""
echo "=== Output folder structure ==="
ls "$RESULTS_DIR/$RUN_NAME/test_latest/images/" 2>/dev/null || echo "WARNING: test_latest/images/ not found"

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/asp/logs/gpu_infer_MIST_e100.csv"

echo ""
echo "MIST inference complete. Next step: sbatch eval_MIST_e100.sh"
