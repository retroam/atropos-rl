"""
MLPerf Optimizer Environment processing script.
This script processes the environment and creates visualization artifacts.
"""

import asyncio
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from atroposlib.frontend.jsonl2html import convert_jsonl_to_html
from atroposlib.utils.io import save_jsonl

# Add the current directory to the path so we can import the environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment import MLPerfOptimizerEnv, MLPerfBenchmark, MLPerfOptimizationRow

async def process_rollouts(env: MLPerfOptimizerEnv, num_examples: int = 20):
    """Process rollouts from the environment and generate visualization data."""
    print("Starting rollout collection...")
    
    rollouts = []
    for _ in range(num_examples):
        item = await env.get_next_item()
        if item:
            trajectories, _ = await env.collect_trajectories(item)
            
            # Store only the necessary information for visualization
            example = {
                "benchmark": item.benchmark.name,
                "query": item.query,
                "responses": []
            }
            
            for rollout in trajectories.rollout_group_data:
                example["responses"].append({
                    "completion": rollout["messages"][-1]["content"],
                    "metrics": rollout["current_metrics"],
                    "target_metrics": rollout["target_metrics"],
                })
            
            scores = await env.score(trajectories)
            if scores:
                for i, score in enumerate(scores["scores"]):
                    if i < len(example["responses"]):
                        example["responses"][i]["score"] = float(score)
            
            rollouts.append(example)
    
    print(f"Collected {len(rollouts)} rollout examples")
    return rollouts

def create_jsonl(rollouts: List[Dict], output_path: str):
    """Create JSONL file from rollouts."""
    examples = []
    
    for rollout in rollouts:
        # Format benchmark information
        benchmark_info = {
            "name": rollout["benchmark"],
            "current_metrics": rollout["responses"][0]["metrics"] if rollout["responses"] else {},
            "target_metrics": rollout["responses"][0]["target_metrics"] if rollout["responses"] else {},
        }
        
        # Format the responses with scores
        responses = []
        for response in rollout["responses"]:
            responses.append({
                "completion": response["completion"],
                "score": response.get("score", 0.0),
            })
        
        # Sort responses by score (best first)
        responses = sorted(responses, key=lambda x: x.get("score", 0), reverse=True)
        
        # Create the example entry
        example = {
            "benchmark": benchmark_info,
            "query": rollout["query"],
            "responses": responses,
        }
        
        examples.append(example)
    
    # Save to JSONL
    save_jsonl(examples, output_path)
    print(f"Saved {len(examples)} examples to {output_path}")
    return output_path

def create_html(jsonl_path: str, output_path: str):
    """Create HTML visualization from JSONL file."""
    convert_jsonl_to_html(jsonl_path, output_path)
    print(f"Generated HTML visualization at {output_path}")
    return output_path

def create_zip(files: List[str], output_path: str):
    """Create ZIP archive of artifacts."""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))
    
    print(f"Created artifact archive at {output_path}")
    return output_path

async def main():
    """Main processing function."""
    print("Initializing MLPerf Optimizer Environment...")
    
    # Initialize environment
    env_config, server_configs = MLPerfOptimizerEnv.config_init()
    
    # Disable WandB for processing
    env_config.use_wandb = False
    
    # Create environment
    env = MLPerfOptimizerEnv(env_config, server_configs, slurm=False, testing=True)
    
    # Setup environment
    await env.setup()
    
    # Create output directory
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    
    # Process rollouts
    rollouts = await process_rollouts(env)
    
    # Create JSONL file
    jsonl_path = str(output_dir / "mlperf_optimizer_examples.jsonl")
    create_jsonl(rollouts, jsonl_path)
    
    # Create HTML visualization
    html_path = str(output_dir / "mlperf_optimizer_visualization.html")
    create_html(jsonl_path, html_path)
    
    # Create ZIP archive
    zip_path = str(output_dir / "mlperf_optimizer_artifacts.zip")
    create_zip([jsonl_path, html_path], zip_path)
    
    print("Processing complete!")
    print(f"Generated artifacts: {zip_path}")

if __name__ == "__main__":
    asyncio.run(main())