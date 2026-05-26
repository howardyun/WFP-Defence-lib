#!/bin/bash
set -e

cd "$(dirname "$0")/palette"

echo "========== Step 1: Extract TAM Features =========="
python extract-list.py

echo "========== Step 2: Website Clustering =========="
python cluster.py

echo "========== Step 3: Super-Matrix Refinement =========="
python refinement.py

echo "========== Step 4: Regularization (Defense Simulation) =========="
python regularization.py

echo "========== Pipeline Complete =========="
