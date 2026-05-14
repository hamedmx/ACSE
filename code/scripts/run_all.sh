#!/bin/bash

set -e

echo "=========================================================="
echo "   ACSE: Adaptive Conformal Semantic Entropy Pipeline    "
echo "=========================================================="

# 1. Environment Verification
echo "[1/5] Checking environment..."
if ! python -c "import torch; import transformers" &> /dev/null; then
    echo "Dependencies missing. Installing from requirements.txt..."
    pip install -r requirements.txt
fi

# 2. Directory Setup
# Ensure folders exist for logs and exported PDFs
echo "[2/5] Initializing workspace..."
mkdir -p logs
mkdir -p results/plots

# 3. Main Pipeline Execution
# Performs Calibration (N=1300) and Inference (N=700) on TriviaQA
echo "[3/5] Running ACSE Main Pipeline (TriviaQA)..."
python main.py \
    --alpha 0.10 \
    --dataset "trivia_qa" \
    --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
    --seed 42 \
    2>&1 | tee logs/main_pipeline.log

# 4. Ablation Studies & Figure Generation
echo "[4/5] Running Ablation Studies and Generating Figures..."
python main.py \
    --alpha 0.10 \
    --dataset "trivia_qa" \
    --run_ablations \
    2>&1 | tee logs/ablations.log

# 5. Summary
echo "[5/5] Finalizing..."
mv *.pdf results/plots/ 2>/dev/null || true

echo "=========================================================="
echo "COMPLETED SUCCESSFULLY."
echo "Results summary: logs/main_pipeline.log"
echo "Visualizations:  results/plots/"
echo "=========================================================="