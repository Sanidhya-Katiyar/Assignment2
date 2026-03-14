#!/usr/bin/env bash
# =============================================================================
# scripts/generate_report.sh
# =============================================================================
# Shell wrapper that runs the full Block 6 pipeline in the correct order:
#   1. Generate PCA + t-SNE visualizations
#   2. Run statistical tests (auto-detect all valid pairs)
#   3. Assemble the results_summary.md and summary_table.csv
#   4. Print the path to the finished report
#
# Usage
# -----
#   bash scripts/generate_report.sh [CONFIG] [OUTDIR]
#
# Defaults
#   CONFIG = configs/baseline.yaml
#   OUTDIR = outputs
#
# Example
#   bash scripts/generate_report.sh configs/baseline.yaml outputs
#
# Notes
# -----
# • All Python scripts are run from the project root.
# • Set PYTHON to point to your virtual-env interpreter if needed:
#     PYTHON=/path/to/venv/bin/python bash scripts/generate_report.sh
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CONFIG="${1:-configs/baseline.yaml}"
OUTDIR="${2:-outputs}"
PYTHON="${PYTHON:-python}"

REPORT_DIR="${OUTDIR}/report"

echo "========================================================"
echo "  AID Transfer Learning — Block 6 Report Generation"
echo "========================================================"
echo "  Config  : ${CONFIG}"
echo "  Output  : ${OUTDIR}"
echo "  Python  : $(${PYTHON} --version 2>&1)"
echo ""

# --------------------------------------------------------------------------- #
# Step 1 — Generate visualizations (PCA + 3 t-SNE runs)
# --------------------------------------------------------------------------- #
echo "── Step 1/4  Generating embeddings and plots ────────────"
${PYTHON} scripts/generate_visualizations.py \
    --config  "${CONFIG}" \
    --models  resnet50 efficientnet_b0 convnext_tiny \
    --layers  early middle final \
    --dr      pca tsne \
    --tsne-runs 3 \
    --outdir  "${OUTDIR}"
echo ""

# --------------------------------------------------------------------------- #
# Step 2 — Statistical tests (auto mode)
# --------------------------------------------------------------------------- #
echo "── Step 2/4  Running statistical tests ──────────────────"
${PYTHON} scripts/run_stat_tests.py \
    --config "${CONFIG}" \
    --auto \
    --outdir "${OUTDIR}"
echo ""

# --------------------------------------------------------------------------- #
# Step 3 — Assemble report
# --------------------------------------------------------------------------- #
echo "── Step 3/4  Assembling report ──────────────────────────"
${PYTHON} src/visualization/report_generator.py \
    --config "${CONFIG}" \
    --out    "${REPORT_DIR}"
echo ""

# --------------------------------------------------------------------------- #
# Step 4 — Print location
# --------------------------------------------------------------------------- #
echo "── Step 4/4  Done ───────────────────────────────────────"
echo ""
echo "  Report   → ${REPORT_DIR}/results_summary.md"
echo "  Table    → ${REPORT_DIR}/summary_table.csv"
echo "  Stats    → ${OUTDIR}/stats/stat_tests.csv"
echo "  Plots    → ${OUTDIR}/plots/"
echo "  Embeddings → ${OUTDIR}/embeddings/"
echo ""
echo "========================================================"
