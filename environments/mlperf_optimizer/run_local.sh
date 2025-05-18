#!/bin/bash
# Script to run the MLPerf Optimizer environment locally

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create directories for output
mkdir -p checkpoints
mkdir -p empty_dir
mkdir -p artifacts

# Step 1: Start the Orchestration Server
echo "Starting Atropos API server..."
cd empty_dir
run-api &
API_PID=$!
cd ..

# Wait for API server to start
echo "Waiting for API server to start..."
sleep 5

# Step 2: Start the Environment
echo "Starting MLPerf Optimizer Environment..."
python3 environment.py serve --slurm false &
ENV_PID=$!

# Wait for environment to start
echo "Waiting for environment to start..."
sleep 10

# Step 3: Start the Trainer
echo "Starting MLPerf Optimizer Trainer..."
python3 trainer.py \
  --model_name "NousResearch/DeepHermes-3-Llama-3-3B-Preview" \
  --training_steps 20 \
  --batch_size 2 \
  --device "cuda" \
  --use_lora \
  --use_wandb \
  --wandb_project "mlperf_optimizer" \
  --save_path "./checkpoints"

# Clean up
echo "Training complete. Cleaning up..."
kill $ENV_PID
kill $API_PID

echo "All done!"