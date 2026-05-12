#!/bin/bash
#SBATCH --job-name=asp_val_MIST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/asp/logs/train_validate_MIST.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/asp/logs/train_validate_MIST.%j.err

# train_validate_MIST.sh
# Runs 3 epochs of ASP training on MIST-HER2 as a cluster confirmation gate.
# This job must pass before submitting the full MIST training jobs.
#
# MIST-HER2.sqsh is used directly -- its top level is already
# trainA/, trainB/, valA/, valB/, which matches what ASP expects.
#
# Submit: sbatch train_validate_MIST.sh
#
# Pass criteria:
#   1. Job exits cleanly (no Python traceback in log)
#   2. Loss values in log are not NaN
#   3. Checkpoint files exist after the job:
#        find $VSC_DATA/projects/asp/outputs/checkpoints/MIST_validate_e3 -name "*.pth"
#   4. GPU log CSV has entries:
#        tail -5 $VSC_DATA/projects/asp/logs/gpu_train_validate_MIST.csv

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/asp_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/asp/code/asp"
CHECKPOINTS_DIR="$VSC_DATA/projects/asp/outputs/checkpoints"
RUN_NAME="MIST_validate_e3"
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
echo "=== SquashFS check ==="
if [ ! -f "$MIST_SQSH" ]; then
    echo "ERROR: MIST-HER2.sqsh not found: $MIST_SQSH"
    exit 1
fi
echo "  MIST-HER2.sqsh found"

echo ""
echo "=== Dataset check ==="
mkdir -p "$MIST_MNT"
apptainer exec \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  trainA: \$(ls $MIST_MNT/trainA | wc -l) images\"; echo \"  trainB: \$(ls $MIST_MNT/trainB | wc -l) images\"; echo \"  valA:   \$(ls $MIST_MNT/valA | wc -l) images\"; echo \"  valB:   \$(ls $MIST_MNT/valB | wc -l) images\""

echo ""
echo "=== Repo check ==="
if [ ! -f "$REPO_DIR/train.py" ]; then
    echo "ERROR: train.py not found in $REPO_DIR"
    exit 1
fi
echo "  train.py found"

mkdir -p "$CHECKPOINTS_DIR/$RUN_NAME"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/asp/logs/gpu_train_validate_MIST.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting validation training (3 epochs, MIST-HER2) ==="
echo "  run name    : $RUN_NAME"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"
echo "  dataroot    : $MIST_MNT (inside MIST-HER2.sqsh)"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    python train.py \
        --dataroot        "$MIST_MNT" \
        --name            "$RUN_NAME" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --model           cpt \
        --CUT_mode        FastCUT \
        --dataset_mode    aligned \
        --direction       AtoB \
        --netG            resnet_6blocks \
        --netD            n_layers \
        --n_layers_D      5 \
        --normG           instance \
        --normD           instance \
        --weight_norm     spectral \
        --no_dropout \
        --nce_layers      0,4,8,12,16 \
        --load_size       1024 \
        --crop_size       512 \
        --preprocess      crop \
        --batch_size      1 \
        --n_epochs        3 \
        --n_epochs_decay  0 \
        --save_epoch_freq 1 \
        --display_id      -1 \
        --num_threads     8 \
        --gpu_ids         0

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID

echo ""
echo "=== Post-run checkpoint check ==="
find "$CHECKPOINTS_DIR/$RUN_NAME" -name "*.pth" | sort

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/asp/logs/gpu_train_validate_MIST.csv"

echo ""
echo "Validation training complete. Review the output above before submitting full runs."
echo "Record time-per-epoch from the log to estimate wall time for the full 100-epoch job."
