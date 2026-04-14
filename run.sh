#!/bin/bash
set -e  # stop immediately if any command fails

echo "=== Installing dependencies ==="
pip install -r requirements.txt -q

echo "=== Downloading dataset ==="
python src/data_pipeline.py

echo "== Initializing CNN Model =="
python src/models/cnn.py

echo "=== Training CNN model ==="
python src/train.py

echo "=== Evaluating model ==="
python src/eval.py

echo "=== Running RL agent ==="
python src/rl_agent.py

echo "=== Running NLP component ==="
python src/models/nlp.py

echo "=== Done! Check experiments/results/ for all plots and metrics ==="
