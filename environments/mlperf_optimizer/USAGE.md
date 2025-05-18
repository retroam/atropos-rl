# MLPerf Optimizer Environment - Usage Guide

This document provides detailed instructions for using the MLPerf Optimizer environment.

## Prerequisites

- Python 3.10 or later
- PyTorch 2.0 or later
- CUDA-capable GPU (for optimal performance)
- Weights & Biases account (for experiment tracking)

## Installation

1. **Clone Atropos Repository**:
   ```bash
   git clone https://github.com/NousResearch/Atropos.git
   cd Atropos
   ```

2. **Install Atropos Dependencies**:
   ```bash
   pip install -e .
   ```

3. **Install MLPerf Optimizer Dependencies**:
   ```bash
   cd environments/hack0/mlperf_optimizer
   pip install -r requirements.txt
   ```

## Running the Environment

### Method 1: Using the Convenience Scripts

We provide shell scripts to easily run the environment:

1. **Run the Local Training Pipeline**:
   ```bash
   cd environments/hack0/mlperf_optimizer
   ./run_local.sh
   ```
   This will:
   - Start the Atropos API server
   - Start the MLPerf Optimizer environment
   - Start the trainer with default parameters
   - Clean up processes when done

2. **Generate Artifacts**:
   ```bash
   ./run_process.sh
   ```
   This will:
   - Run the processing script to generate examples
   - Create JSONL and HTML visualization files
   - Package them into a ZIP archive

### Method 2: Manual Execution

For more control over the execution process:

1. **Start the Orchestration Server**:
   ```bash
   mkdir -p empty_dir && cd empty_dir
   run-api
   ```

2. **Start the Environment** (in a new terminal):
   ```bash
   cd /path/to/Atropos
   python environments/hack0/mlperf_optimizer/environment.py serve --slurm false
   ```

3. **Start the Trainer** (in a third terminal):
   ```bash
   cd /path/to/Atropos/environments/hack0/mlperf_optimizer
   python trainer.py --model_name "NousResearch/DeepHermes-3-Llama-3-3B-Preview" --training_steps 500 --batch_size 4 --use_lora --use_wandb
   ```

4. **Generate Artifacts** (after training):
   ```bash
   python process.py
   ```

## Configuration Options

### Environment Configuration

You can modify the environment behavior by editing `config.yaml` or passing command-line arguments to the trainer.

Key configuration options:
- `model_name`: Base model to use (default: "NousResearch/DeepHermes-3-Llama-3-3B-Preview")
- `training_steps`: Number of training steps to perform
- `batch_size`: Batch size for training
- `use_lora`: Whether to use LoRA for parameter-efficient fine-tuning
- `device`: Training device ("cuda" or "cpu")

### Using a Different Model

To use a different base model:

1. Edit the model name in `environment.py` under the `config_init` method:
   ```python
   env_config = BaseEnvConfig(
       tokenizer_name="YOUR_MODEL_NAME",
       # other settings...
   )
   ```

2. Also update the model name in `trainer.py` when running:
   ```bash
   python trainer.py --model_name "YOUR_MODEL_NAME"
   ```

## Understanding the Output

### Training Metrics

During training, the following metrics are logged:
- **Loss**: Overall training loss
- **PG Loss**: Policy gradient loss component
- **KL Loss**: KL divergence loss component
- **Learning Rate**: Current learning rate

### WandB Integration

If you enable WandB logging, you'll see:
- **Optimization Quality**: How effective the suggestions are
- **Predicted Improvement**: Estimated performance gain
- **Solution Feasibility**: How implementable the suggestions are
- **Completion Length**: Tracking solution conciseness

### Artifacts

After running `process.py`, you'll get:
- **JSONL File**: Raw examples with queries and completions
- **HTML Visualization**: Interactive view of optimization suggestions
- **ZIP Archive**: Compressed package of both files

## Extending the Environment

To extend the MLPerf Optimizer environment:

1. **Add New Benchmarks**:
   - Modify the `_create_synthetic_dataset` method in `environment.py`
   - Add new `MLPerfBenchmark` entries with appropriate metrics

2. **Customize Reward Function**:
   - Adjust the scoring methods in `environment.py`:
     - `_rate_optimization_quality`
     - `_rate_solution_feasibility`
     - `_predict_metric_improvement`

3. **Integrate Real Benchmarks**:
   - Replace the synthetic dataset with real MLPerf benchmark data
   - Implement simulation-based evaluation of suggested optimizations

## Troubleshooting

### Common Issues

1. **vLLM Server Fails to Start**:
   - Check GPU memory availability
   - Verify that the model is available from HuggingFace
   - Try using a smaller model

2. **Training Doesn't Progress**:
   - Check that all components are running (API server, environment, trainer)
   - Verify network connectivity between components
   - Check for errors in the logs

3. **Out of Memory Errors**:
   - Reduce batch size
   - Enable LoRA (--use_lora)
   - Increase gradient accumulation steps

### Getting Help

If you encounter issues not covered here, please:
1. Check the Atropos documentation
2. Open an issue on the Atropos GitHub repository
3. Contact the hackathon organizers for assistance