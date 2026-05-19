#!/bin/bash
#SBATCH --job-name=asp_train_MIST-HER2_e100_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/asp/logs/train_full_MIST-HER2_e100_p1.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/asp/logs/train_full_MIST-HER2_e100_p1.%j.err

# train_MIST-HER2_full_e100_part1.sh
# Epochs 1-50 of ASP training on MIST-HER2 at 512x512 (cropped from 1024).
# Constant LR throughout (n_epochs_decay=0).
#
# MIST has 4642 training images. Use time-per-epoch from the validate job log to
# set the wall time above: (sec_per_epoch * 50 * 1.20) / 3600 rounded up to next hour.
#
# DO NOT submit this manually -- use submit_MIST-HER2_full_e100.sh which chains both parts.
#
# Checkpoints saved at epoch 25 and epoch 50 (plus latest after every epoch):
#   $VSC_DATA/projects/asp/outputs/checkpoints/MIST-HER2_full_e100/

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/asp_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/asp/code/asp"
CHECKPOINTS_DIR="$VSC_DATA/projects/asp/outputs/checkpoints"
RUN_NAME="MIST-HER2_full_e100"
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
echo "=== Dataset check ==="
mkdir -p "$MIST_MNT"
apptainer exec \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  trainA: \$(ls $MIST_MNT/trainA | wc -l) images\"; echo \"  trainB: \$(ls $MIST_MNT/trainB | wc -l) images\""

mkdir -p "$CHECKPOINTS_DIR/$RUN_NAME"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/asp/logs/gpu_train_full_MIST-HER2_e100_p1.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting MIST training part 1 (epochs 1-50, constant LR) ==="
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
        --n_epochs        50 \
        --n_epochs_decay  0 \
        --save_epoch_freq 25 \
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
tail -3 "$VSC_DATA/projects/asp/logs/gpu_train_full_MIST-HER2_e100_p1.csv"

echo ""
echo "MIST part 1 complete (epochs 1-50). Part 2 should start automatically if submitted via wrapper."
