#!/bin/bash
#SBATCH --job-name=asp_MIST_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=XX:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/asp/logs/train_MIST_p2.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/asp/logs/train_MIST_p2.%j.err

# train_MIST_e100_part2.sh
# Epochs 51-100 of ASP training on MIST-HER2 at 512x512.
# Resumes from the latest checkpoint saved by part 1.
# Linear LR decay applied across these 50 epochs (n_epochs_decay=50).
#
# The LR schedule is equivalent to a single 100-epoch run:
#   Part 1: epochs  1-50  constant LR  (n_epochs=50, n_epochs_decay=0)
#   Part 2: epochs 51-100 linear decay (epoch_count=51, n_epochs=50, n_epochs_decay=50)
#
# DO NOT submit this manually before part 1 finishes -- use submit_MIST_e100.sh.
# If part 1 failed or was cancelled, do not submit this script.
#
# After this job completes, next step: sbatch infer_MIST_e100.sh

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/asp_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/asp/code/asp"
CHECKPOINTS_DIR="$VSC_DATA/projects/asp/outputs/checkpoints"
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
echo "=== Dataset check ==="
mkdir -p "$MIST_MNT"
apptainer exec \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  trainA: \$(ls $MIST_MNT/trainA | wc -l) images\"; echo \"  trainB: \$(ls $MIST_MNT/trainB | wc -l) images\""

echo ""
echo "=== Checkpoint check (part 1 must have completed) ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -f "$CKPT_DIR/latest_net_G.pth" ]; then
    echo "ERROR: latest_net_G.pth not found in $CKPT_DIR"
    echo "Has part 1 completed successfully?"
    exit 1
fi
echo "  latest checkpoint found:"
find "$CKPT_DIR" -name "latest_net_*.pth" | sort

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/asp/logs/gpu_train_MIST_p2.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting MIST training part 2 (epochs 51-100, LR decay) ==="
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
        --n_epochs_decay  50 \
        --epoch_count     51 \
        --continue_train \
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
tail -3 "$VSC_DATA/projects/asp/logs/gpu_train_MIST_p2.csv"

echo ""
echo "MIST full training complete (epochs 1-100). Next step: sbatch infer_MIST_e100.sh"
