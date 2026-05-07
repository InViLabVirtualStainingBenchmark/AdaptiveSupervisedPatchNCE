#!/bin/bash
#SBATCH --job-name=asp_eval_MIST_e100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/asp/logs/eval_MIST_e100.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/asp/logs/eval_MIST_e100.%j.err

# eval_MIST_e100.sh
# Runs evaluate.py on ASP MIST-HER2 inference outputs using the shared evaluate_nvidia.sif.
#
# GT images come from valB inside MIST-HER2.sqsh.
# Predictions come from the inference output folder (on $VSC_DATA, no sqsh needed).
#
# Submit ONLY after infer_MIST_e100.sh has completed and image count is correct.
# Submit: sbatch eval_MIST_e100.sh
#
# Results appended to:
#   $VSC_DATA/benchmark_results.csv

set -euo pipefail

EVAL_CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"
RESULTS_DIR="$VSC_DATA/projects/asp/outputs/results"
RUN_NAME="MIST_e100"
PRED_DIR="$RESULTS_DIR/$RUN_NAME/test_latest/images/fake_B"
MIST_SQSH="$VSC_SCRATCH/MIST-HER2.sqsh"
MIST_MNT="$VSC_SCRATCH/sqsh_mnt/MIST-HER2"
GT_DIR="$MIST_MNT/valB"

# =========================
# MODULES
# =========================

module purge
module load calcua/2026.1

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Evaluate container ==="
if [ ! -f "$EVAL_CONTAINER" ]; then
    echo "ERROR: evaluate_nvidia.sif not found: $EVAL_CONTAINER"
    exit 1
fi
echo "  found"

echo ""
echo "=== SquashFS check ==="
if [ ! -f "$MIST_SQSH" ]; then
    echo "ERROR: MIST-HER2.sqsh not found: $MIST_SQSH"
    exit 1
fi
echo "  MIST-HER2.sqsh found"

echo ""
echo "=== Prediction folder check ==="
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: Prediction folder not found: $PRED_DIR"
    echo "Has infer_MIST_e100.sh completed successfully?"
    exit 1
fi
echo "  fake_B images: $(find "$PRED_DIR" -name "*.jpg" | wc -l)"

# =========================
# EVALUATION
# =========================

mkdir -p "$MIST_MNT"

echo ""
echo "=== Starting MIST evaluation ==="
echo "  predictions : $PRED_DIR"
echo "  ground truth: $GT_DIR (inside MIST-HER2.sqsh)"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$EVAL_CONTAINER" \
    python "$VSC_DATA/evaluate/evaluate.py" \
        --pred         "$PRED_DIR" \
        --gt           "$GT_DIR" \
        --model_name   ASP \
        --dataset_name MIST-HER2 \
        --split_name   full_e100 \
        --match_by     stem \
        --output       "$VSC_DATA/benchmark_results.csv"

# =========================
# POST-RUN REPORT
# =========================

echo ""
echo "=== benchmark_results.csv (last 3 rows) ==="
tail -3 "$VSC_DATA/benchmark_results.csv"

echo ""
echo "MIST evaluation complete."
