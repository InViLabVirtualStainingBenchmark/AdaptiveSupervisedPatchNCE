#!/bin/bash
#SBATCH --job-name=asp_val_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/asp/logs/train_validate_BCI.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/asp/logs/train_validate_BCI.%j.err

# train_validate_BCI.sh
# Runs 3 epochs of ASP training on BCI as a cluster confirmation gate.
# This job must pass before submitting the full training jobs.
#
# PREREQUISITE -- BCI-asp.sqsh:
#   ASP uses --dataset_mode aligned with separate trainA/trainB/testA/testB folders.
#   The generic BCI.sqsh has HE/IHC layout, which does not match.
#   If BCI-asp.sqsh does not already exist on scratch, create it once on the login node:
#
#     mkdir -p $VSC_SCRATCH/bci_asp_staging/{trainA,trainB,testA,testB}
#     unsquashfs -d $VSC_SCRATCH/bci_unsq $VSC_SCRATCH/BCI.sqsh
#     cp $VSC_SCRATCH/bci_unsq/HE/train/* $VSC_SCRATCH/bci_asp_staging/trainA/
#     cp $VSC_SCRATCH/bci_unsq/IHC/train/* $VSC_SCRATCH/bci_asp_staging/trainB/
#     cp $VSC_SCRATCH/bci_unsq/HE/test/*  $VSC_SCRATCH/bci_asp_staging/testA/
#     cp $VSC_SCRATCH/bci_unsq/IHC/test/* $VSC_SCRATCH/bci_asp_staging/testB/
#     mksquashfs $VSC_SCRATCH/bci_asp_staging $VSC_SCRATCH/BCI-asp.sqsh -noappend
#     rm -rf $VSC_SCRATCH/bci_unsq $VSC_SCRATCH/bci_asp_staging
#
# Submit: sbatch train_validate_BCI.sh
#
# Pass criteria:
#   1. Job exits cleanly (no Python traceback in log)
#   2. Loss values in log are not NaN
#   3. Checkpoint files exist after the job:
#        find $VSC_DATA/projects/asp/outputs/checkpoints/BCI_validate_e3 -name "*.pth"
#   4. GPU log CSV has entries:
#        tail -5 $VSC_DATA/projects/asp/logs/gpu_train_validate_BCI.csv

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/asp_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/asp/code/asp"
CHECKPOINTS_DIR="$VSC_DATA/projects/asp/outputs/checkpoints"
RUN_NAME="BCI_validate_e3"
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
echo "=== SquashFS check ==="
if [ ! -f "$BCI_ASP_SQSH" ]; then
    echo "ERROR: BCI-asp.sqsh not found: $BCI_ASP_SQSH"
    echo "See the PREREQUISITE section at the top of this script."
    exit 1
fi
echo "  BCI-asp.sqsh found"

echo ""
echo "=== Dataset check ==="
mkdir -p "$BCI_ASP_MNT"
apptainer exec \
    -B "$BCI_ASP_SQSH:$BCI_ASP_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  trainA: \$(ls $BCI_ASP_MNT/trainA | wc -l) images\"; echo \"  trainB: \$(ls $BCI_ASP_MNT/trainB | wc -l) images\"; echo \"  valA:   \$(ls $BCI_ASP_MNT/valA   | wc -l) images\"; echo \"  valB:   \$(ls $BCI_ASP_MNT/valB   | wc -l) images\""

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
    > "$VSC_DATA/projects/asp/logs/gpu_train_validate_BCI.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting validation training (3 epochs, BCI) ==="
echo "  run name    : $RUN_NAME"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"
echo "  dataroot    : $BCI_ASP_MNT (inside BCI-asp.sqsh)"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$BCI_ASP_SQSH:$BCI_ASP_MNT:image-src=/" \
    "$CONTAINER" \
    python train.py \
        --dataroot        "$BCI_ASP_MNT" \
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
tail -3 "$VSC_DATA/projects/asp/logs/gpu_train_validate_BCI.csv"

echo ""
echo "Validation training complete. Review the output above before submitting full runs."
echo "Record time-per-epoch from the log to estimate wall time for the full 100-epoch job."
