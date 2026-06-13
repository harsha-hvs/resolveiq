#!/usr/bin/env bash
# ============================================================
# ResolveIQ — Complete pipeline runner for Tillicum H200
#
# Usage:
#   export TOGETHER_API_KEY=your_key
#   export WANDB_API_KEY=your_key
#   chmod +x run_pipeline.sh && ./run_pipeline.sh
#
# OR run interactively step-by-step via the notebook:
#   jupyter notebook ResolveIQ_IMT526.ipynb
# ============================================================
set -e

echo "======================================================"
echo "  ResolveIQ Pipeline — IMT 526 · Harsh Vardhan"
echo "======================================================"

# ── Guard: check keys ────────────────────────────────────────
[ -z "$TOGETHER_API_KEY" ] && echo "ERROR: set TOGETHER_API_KEY" && exit 1
[ -z "$WANDB_API_KEY" ]    && echo "ERROR: set WANDB_API_KEY"    && exit 1

# ── 0. Install ───────────────────────────────────────────────
echo "[0] Installing packages..."
pip install -q transformers==4.47.0 peft==0.14.0 trl==1.1.0 \
    bitsandbytes==0.44.1 together==1.3.0 bert-score==0.3.13 \
    datasets==3.1.0 accelerate==1.2.0 wandb sentencepiece \
    matplotlib nbformat

# ── 1. Data download ─────────────────────────────────────────
echo "[1] Data download..."
mkdir -p data checkpoints results serve
if [ ! -f "data/Posts.xml" ]; then
    [ ! -f "data/serverfault.com.7z" ] && \
        wget -q --show-progress \
        "https://archive.org/download/stackexchange/serverfault.com.7z" \
        -O data/serverfault.com.7z
    sudo apt-get install -y p7zip-full -qq
    7z e data/serverfault.com.7z Posts.xml -odata/ -y
fi
echo "✓ Posts.xml ready"

# ── 2. Build splits ──────────────────────────────────────────
echo "[2] Building dataset splits..."
[ ! -f "data/train.jsonl" ] && \
    python data/build_dataset.py --xml data/Posts.xml --out data/ --n 1500
echo "✓ Splits ready"

# ── 3. Synthetic targets ─────────────────────────────────────
echo "[3] Generating synthetic targets via Together AI (~45 min)..."
for split in train val test; do
    [ ! -f "data/${split}_labeled.jsonl" ] && \
        python data/gen_targets.py --split $split --data data/
    n=$(wc -l < "data/${split}_labeled.jsonl")
    echo "  ✓ ${split}: $n examples"
done

# ── 4. SFT ───────────────────────────────────────────────────
echo "[4] QLoRA SFT (~13-20 min on H200)..."
[ ! -d "checkpoints/sft_v1" ] && python train/sft.py
echo "✓ SFT complete"

# ── 5. Baseline evaluation (professor feedback item #2) ──────
echo "[5] Zero-shot baseline evaluation (~15 min)..."
[ ! -f "results/eval_baseline.json" ] && \
    python eval/evaluate.py \
        --models baseline \
        --data data/test_labeled.jsonl \
        --out results/eval_baseline.json
echo "✓ Baseline logged"

# ── 6. DPO ───────────────────────────────────────────────────
echo "[6] DPO verbosity pass..."
[ ! -f "data/dpo_pairs.jsonl" ] && python train/gen_dpo_rejects.py
[ ! -d "checkpoints/dpo_v1"  ] && python train/dpo.py
[ ! -d "checkpoints/merged_final" ] && python train/merge.py
echo "✓ DPO + merge complete"

# ── 7. Full evaluation ───────────────────────────────────────
echo "[7] Full ablation evaluation (~45 min)..."
python eval/evaluate.py \
    --models baseline post_sft dpo_merged \
    --data data/test_labeled.jsonl \
    --out results/eval.json

# ── 8. LIMA hedge (optional — set to 1 to run) ───────────────
RUN_LIMA=0
if [ "$RUN_LIMA" = "1" ]; then
    echo "[8] LIMA learning curve (~52 min)..."
    python train/sft_subset.py --sizes 200 500 1000
    python eval/learning_curve.py
fi

echo ""
echo "======================================================"
echo "  PIPELINE COMPLETE"
echo "======================================================"
echo "  Ablation table:  results/eval.json"
echo "  Chart:           results/ablation_chart.png"
echo "  Predictions:     results/*_predictions.jsonl"
echo "  Merged model:    checkpoints/merged_final/"
echo ""
echo "  Next: open ResolveIQ_IMT526.ipynb for §G (manual inspection)"
echo "        and §K (report checklist)"
echo "======================================================"
