import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from atroposlib.envs.base import BaseEnv, BaseEnvConfig
from atroposlib.envs.server_handling.server_harness import APIServerConfig
from atroposlib.type_definitions import Item, ScoredDataGroup
from datasets import load_dataset
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# System prompt for the MLPerf optimization tasks
SYSTEM_PROMPT = """You are an AI assistant specialized in optimizing machine learning models for benchmark performance.
Your task is to suggest optimizations to improve model performance on MLPerf benchmarks.
Focus on providing concrete, implementable suggestions that can measurably improve inference speed,
training throughput, or accuracy metrics while maintaining model quality.
Base your suggestions on proven ML optimization techniques and explain the reasoning behind each suggestion.
"""

@dataclass
class MLPerfBenchmark:
    """Data structure representing a MLPerf benchmark task."""
    name: str
    description: str
    current_implementation: str
    metrics: Dict[str, float]
    target_metrics: Dict[str, float]
    constraints: List[str]

@dataclass
class MLPerfOptimizationRow:
    """Data structure representing a row in our optimization dataset."""
    benchmark: MLPerfBenchmark
    query: str
    reference_solution: str
    
class MLPerfOptimizerEnv(BaseEnv):
    """Environment for optimizing ML models for MLPerf benchmarks."""
    name = "mlperf_optimizer"
    
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        # Metrics tracking
        self.optimization_score_buffer = list()
        self.metrics_improvement_buffer = list()
        self.solution_quality_buffer = list()
        self.eval_metrics = list()
        # WandB visualization data
        self.rollouts_for_wandb = []
        self.completion_lengths = []
    
    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        """Initialize configuration for the environment and API server."""
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,  # Number of rollouts per query
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=8,
            steps_per_eval=50,
            max_token_length=2048,
            wandb_name="mlperf_optimizer",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",  # Points to vLLM server started by trainer
                api_key="x",  # Placeholder, vLLM doesn't require API key
                num_requests_for_eval=128,
            ),
        ]
        return env_config, server_configs
    
    async def setup(self):
        """Load and preprocess MLPerf benchmark optimization dataset."""
        # For the hackathon, we'll create a synthetic dataset
        # In a real implementation, we'd load from a real dataset
        self.benchmarks = self._create_synthetic_dataset()
        self.train_data = self._create_train_data(self.benchmarks)
        self.eval_data = self._create_eval_data(self.benchmarks)
        self.iter = 0
    
    def _create_synthetic_dataset(self):
        """Create a synthetic dataset of MLPerf benchmark tasks."""
        benchmarks = []
        
        # Image Classification benchmark
        benchmarks.append(MLPerfBenchmark(
            name="ResNet50 Image Classification",
            description="Optimize ResNet50 for image classification on ImageNet dataset",
            current_implementation="""
import torch
import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def inference(model, input_batch):
    with torch.no_grad():
        output = model(input_batch)
    return output
""",
            metrics={
                "accuracy": 0.762,
                "inference_time_ms": 87.4,
                "memory_usage_mb": 98.2,
            },
            target_metrics={
                "accuracy": 0.762,  # Maintain accuracy
                "inference_time_ms": 70.0,  # Target 20% speedup
                "memory_usage_mb": 85.0,  # Target 15% memory reduction
            },
            constraints=[
                "Must maintain at least 99.5% of current accuracy",
                "Cannot use quantization below INT8",
                "Must work on standard GPU hardware",
            ]
        ))
        
        # BERT NLP benchmark
        benchmarks.append(MLPerfBenchmark(
            name="BERT Question Answering",
            description="Optimize BERT model for question answering on SQuAD dataset",
            current_implementation="""
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

def get_model():
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model.eval()
    return model, tokenizer

def inference(model, tokenizer, question, context):
    inputs = tokenizer(question, context, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end+1])
    return answer
""",
            metrics={
                "f1_score": 0.881,
                "inference_time_ms": 132.6,
                "memory_usage_mb": 435.8,
            },
            target_metrics={
                "f1_score": 0.881,  # Maintain F1 score
                "inference_time_ms": 100.0,  # Target 25% speedup
                "memory_usage_mb": 350.0,  # Target 20% memory reduction
            },
            constraints=[
                "Must maintain at least 99% of current F1 score",
                "Must work with standard PyTorch installation",
                "Cannot use specialized hardware (TPUs, etc.)",
            ]
        ))
        
        # Object Detection benchmark
        benchmarks.append(MLPerfBenchmark(
            name="SSD Object Detection",
            description="Optimize SSD MobileNet for object detection on COCO dataset",
            current_implementation="""
import torch
import torchvision

def get_model():
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()
    return model

def inference(model, images):
    with torch.no_grad():
        predictions = model(images)
    return predictions
""",
            metrics={
                "mAP": 0.253,
                "inference_time_ms": 112.5,
                "memory_usage_mb": 210.4,
            },
            target_metrics={
                "mAP": 0.253,  # Maintain mAP
                "inference_time_ms": 85.0,  # Target 25% speedup
                "memory_usage_mb": 175.0,  # Target 15% memory reduction
            },
            constraints=[
                "Must maintain at least 98% of current mAP",
                "Must use standard PyTorch operations",
                "Cannot use external libraries beyond PyTorch and torchvision",
            ]
        ))
        
        return benchmarks
    
    def _create_train_data(self, benchmarks):
        """Create training queries from benchmark data."""
        train_data = []
        
        for benchmark in benchmarks:
            # Create different query types for each benchmark
            
            # General optimization query
            train_data.append(MLPerfOptimizationRow(
                benchmark=benchmark,
                query=f"""I'm working with a {benchmark.name} model that needs optimization. 
Here's my current implementation:

```python
{benchmark.current_implementation}
```

Current metrics:
- {', '.join([f'{k}: {v}' for k, v in benchmark.metrics.items()])}

Target metrics:
- {', '.join([f'{k}: {v}' for k, v in benchmark.target_metrics.items()])}

Constraints:
- {' '.join(benchmark.constraints)}

What specific optimizations can I apply to improve the performance while meeting the constraints?""",
                reference_solution=f"Reference solution for {benchmark.name} optimization",
            ))
            
            # Inference speed optimization
            train_data.append(MLPerfOptimizationRow(
                benchmark=benchmark,
                query=f"""I need to optimize the inference speed of my {benchmark.name} model.
Current implementation:

```python
{benchmark.current_implementation}
```

Current inference time: {benchmark.metrics.get('inference_time_ms', 'N/A')}ms
Target inference time: {benchmark.target_metrics.get('inference_time_ms', 'N/A')}ms

How can I make this model run faster while maintaining similar accuracy?""",
                reference_solution=f"Reference solution for {benchmark.name} inference optimization",
            ))
            
            # Memory optimization
            train_data.append(MLPerfOptimizationRow(
                benchmark=benchmark,
                query=f"""My {benchmark.name} model is using too much memory.
Current implementation:

```python
{benchmark.current_implementation}
```

Current memory usage: {benchmark.metrics.get('memory_usage_mb', 'N/A')}MB
Target memory usage: {benchmark.target_metrics.get('memory_usage_mb', 'N/A')}MB

How can I reduce the memory footprint while maintaining model performance?""",
                reference_solution=f"Reference solution for {benchmark.name} memory optimization",
            ))
        
        return train_data
    
    def _create_eval_data(self, benchmarks):
        """Create evaluation queries from benchmark data."""
        # For the hackathon, we'll use a subset of training data for evaluation
        # In a real implementation, we'd create separate evaluation data
        return self.train_data[:len(benchmarks)]
    
    async def get_next_item(self) -> Optional[MLPerfOptimizationRow]:
        """Get the next item for rollout collection."""
        if self.iter >= len(self.train_data):
            self.iter = 0  # Reset to beginning of dataset
        
        result = self.train_data[self.iter]
        self.iter += 1
        return result
    
    async def collect_trajectories(
        self, item: MLPerfOptimizationRow
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        """Collect optimization suggestions from the model."""
        # Prepare the query as a user message
        user_message = {"role": "user", "content": item.query}
        
        # Get multiple completions from the LLM
        chat_completions = await self.server.chat_completion(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, user_message],
            n=self.config.group_size,  # Number of completions to generate
            max_tokens=self.config.max_token_length,
            temperature=0.7,  # Use some temperature for diverse solutions
        )
        
        # Process completions
        items = []
        rollouts = []
        
        for i, completion in enumerate(chat_completions.choices):
            rollout = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    user_message,
                    {"role": "assistant", "content": completion.message.content},
                ],
                "benchmark_name": item.benchmark.name,
                "current_metrics": item.benchmark.metrics,
                "target_metrics": item.benchmark.target_metrics,
            }
            
            rollouts.append(rollout)
            items.append(
                Item(dict={"completion": completion.message.content, "index": i})
            )
        
        to_postprocess = ScoredDataGroup(
            rollout_group_data=rollouts,
            item_data={"reference": item.reference_solution, "query": item.query},
        )
        
        to_backlog = list()  # We're not using backlog in this example
        
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Score optimization suggestions based on quality and predicted improvements."""
        
        scores = {"scores": [], "ids": []}
        completion_lengths = []
        
        for i, rollout in enumerate(rollout_group_data["rollout_group_data"]):
            # Extract completion from the rollout
            completion = rollout["messages"][-1]["content"]
            
            # Calculate metrics for scoring
            optimization_quality = self._rate_optimization_quality(completion)
            feasibility_score = self._rate_solution_feasibility(completion)
            predicted_improvement = self._predict_metric_improvement(
                rollout["benchmark_name"], 
                completion,
                rollout["current_metrics"],
                rollout["target_metrics"]
            )
            
            # Combine scores (higher is better)
            combined_score = (
                0.4 * optimization_quality +
                0.3 * feasibility_score +
                0.3 * predicted_improvement
            )
            
            # Apply length penalty to encourage concise answers
            completion_length = len(completion)
            completion_lengths.append(completion_length)
            length_penalty = max(0, 1.0 - (completion_length - 500) / 2000) if completion_length > 500 else 1.0
            final_score = combined_score * length_penalty
            
            # Store in scores
            scores["scores"].append(final_score)
            scores["ids"].append(i)

            # Log to wandb if enabled
            if self.config.use_wandb:
                self.optimization_score_buffer.append(optimization_quality)
                self.metrics_improvement_buffer.append(predicted_improvement)
                self.solution_quality_buffer.append(feasibility_score)
                self.completion_lengths.append(completion_length)
        
        # Check if scores are all the same (uninformative)
        if len(set(scores["scores"])) <= 1:
            return None
        
        # Prepare all data for scoring
        # Tokenize completions for the trainer
        tokenized_completions = []
        
        for rollout in rollout_group_data["rollout_group_data"]:
            completion = rollout["messages"][-1]["content"]
            # In a real implementation, we'd use tokenizer here, just simulating for the hackathon
            tokenized_completions.append({"input_ids": list(range(100)), "attention_mask": [1] * 100})
        
        scores["message_tokenizations"] = tokenized_completions
        
        # Track metrics for wandb if enabled
        if self.config.use_wandb and len(self.optimization_score_buffer) >= 100:
            wandb.log(
                {
                    "optimization_quality": np.mean(self.optimization_score_buffer),
                    "predicted_improvement": np.mean(self.metrics_improvement_buffer),
                    "solution_feasibility": np.mean(self.solution_quality_buffer),
                    "completion_length": np.mean(self.completion_lengths),
                }
            )
            # Reset buffers
            self.optimization_score_buffer = []
            self.metrics_improvement_buffer = []
            self.solution_quality_buffer = []
            self.completion_lengths = []
        
        return scores

    def _rate_optimization_quality(self, completion: str) -> float:
        """Rate the quality of optimization suggestions.
        
        For the hackathon, we'll use a simple heuristic approach:
        - Check for concrete code examples
        - Check for explanations of benefits
        - Check for handling of constraints
        
        In a real implementation, we'd use a more sophisticated approach.
        """
        score = 0.5  # Base score
        
        # Check for code examples
        if "```python" in completion or "```py" in completion:
            score += 0.2
        
        # Check for explanations
        if "benefit" in completion.lower() or "improve" in completion.lower() or "advantage" in completion.lower():
            score += 0.1
        
        # Check for constraint handling
        if "constraint" in completion.lower() or "requirement" in completion.lower() or "maintain" in completion.lower():
            score += 0.1
        
        # Check for specific optimization techniques
        techniques = ["quantization", "pruning", "distillation", "caching", "fusion", "parallelism", "batching"]
        for technique in techniques:
            if technique in completion.lower():
                score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0

    def _rate_solution_feasibility(self, completion: str) -> float:
        """Rate how feasible/implementable the optimization suggestions are.
        
        For the hackathon, we'll use a simple heuristic approach.
        In a real implementation, we'd use a more sophisticated approach.
        """
        score = 0.5  # Base score
        
        # Check for concrete implementation details
        if "import" in completion or "def" in completion or "class" in completion:
            score += 0.2
        
        # Check for explanation of implementation steps
        if "step" in completion.lower() or "first" in completion.lower() or "then" in completion.lower():
            score += 0.1
        
        # Check for specificity
        if re.search(r"\d+", completion):  # Contains numbers (likely specific parameters)
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

    def _predict_metric_improvement(self, benchmark_name: str, completion: str, 
                                  current_metrics: Dict[str, float],
                                  target_metrics: Dict[str, float]) -> float:
        """Predict how much the suggested optimizations would improve metrics.
        
        For the hackathon, we'll use a simple heuristic approach.
        In a real implementation, we'd use a more sophisticated approach or simulator.
        """
        # Calculate how far current metrics are from targets
        improvements_needed = {}
        for metric, current in current_metrics.items():
            if metric in target_metrics:
                if "time" in metric.lower() or "memory" in metric.lower():
                    # For these metrics, lower is better
                    improvements_needed[metric] = max(0, (current - target_metrics[metric]) / current)
                else:
                    # For accuracy metrics, higher is better
                    improvements_needed[metric] = max(0, (target_metrics[metric] - current) / current)
        
        # Check for techniques that might address specific metrics
        improvement_score = 0.3  # Base score
        
        # Inference speed improvement techniques
        speed_techniques = ["fusion", "jit", "torch.compile", "parallel", "cache", "batch", "thread", "worker", 
                           "quantiz", "int8", "fp16", "half", "prune", "distill"]
        
        # Memory usage improvement techniques
        memory_techniques = ["checkpoint", "offload", "free", "del ", "garbage", "quantiz", "int8", "fp16", "half", 
                            "prune", "activation", "recomput"]
        
        # Accuracy preservation techniques
        accuracy_techniques = ["calibration", "fine-tun", "knowledge distill", "teacher", "ensemble", 
                              "post-train", "bias correction"]
        
        # Count mentions of techniques relevant to the needed improvements
        for metric, needed in improvements_needed.items():
            if needed > 0:
                if any(t in metric.lower() for t in ["time", "latency", "throughput", "speed"]):
                    for technique in speed_techniques:
                        if technique in completion.lower():
                            improvement_score += 0.05
                
                elif any(t in metric.lower() for t in ["memory", "ram", "vram", "gpu"]):
                    for technique in memory_techniques:
                        if technique in completion.lower():
                            improvement_score += 0.05
                
                elif any(t in metric.lower() for t in ["accuracy", "f1", "map", "precision", "recall"]):
                    # For accuracy, we care about maintaining it while improving other metrics
                    for technique in accuracy_techniques:
                        if technique in completion.lower():
                            improvement_score += 0.05
        
        return min(improvement_score, 1.0)  # Cap at 1.0

    async def evaluate(self):
        """Run evaluation on the test set."""
        if not self.eval_data:
            return
        
        eval_scores = []
        
        for item in self.eval_data:
            user_message = {"role": "user", "content": item.query}
            
            # Get a single completion for evaluation
            chat_completion = await self.server.chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, user_message],
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.2,  # Lower temperature for more reliable outputs
            )
            
            completion = chat_completion.choices[0].message.content
            
            # Calculate metrics
            optimization_quality = self._rate_optimization_quality(completion)
            feasibility_score = self._rate_solution_feasibility(completion)
            predicted_improvement = self._predict_metric_improvement(
                item.benchmark.name,
                completion,
                item.benchmark.metrics,
                item.benchmark.target_metrics
            )
            
            # Combined evaluation score
            combined_score = (
                0.4 * optimization_quality +
                0.3 * feasibility_score +
                0.3 * predicted_improvement
            )
            
            eval_scores.append(combined_score)
            
            # Save example to WandB
            if self.config.use_wandb:
                self.rollouts_for_wandb.append({
                    "query": item.query,
                    "completion": completion,
                    "optimization_quality": optimization_quality,
                    "feasibility_score": feasibility_score,
                    "predicted_improvement": predicted_improvement,
                    "combined_score": combined_score,
                })
        
        # Calculate average evaluation score
        avg_eval_score = np.mean(eval_scores)
        
        # Log evaluation metrics
        eval_metrics = {
            "eval/avg_score": avg_eval_score,
        }
        
        if self.config.use_wandb:
            # Log the evaluation metrics and example rollouts
            wandb.log(eval_metrics)
            
            # Log detailed examples (limit to 10 to avoid too much data)
            for i, example in enumerate(self.rollouts_for_wandb[:10]):
                wandb.log({
                    f"eval/example_{i}/query": example["query"],
                    f"eval/example_{i}/completion": example["completion"],
                    f"eval/example_{i}/optimization_quality": example["optimization_quality"],
                    f"eval/example_{i}/feasibility_score": example["feasibility_score"],
                    f"eval/example_{i}/predicted_improvement": example["predicted_improvement"],
                    f"eval/example_{i}/combined_score": example["combined_score"],
                })
            
            # Clear the buffer
            self.rollouts_for_wandb = []
        
        self.eval_metrics.append(eval_metrics)
        return eval_metrics

# Entry point for running the environment directly
if __name__ == "__main__":
    import argparse
    from atroposlib.utils.cli import serve_arguments_register
    
    parser = argparse.ArgumentParser(description="MLPerf Optimizer Environment")
    serve_arguments_register(parser)
    args = parser.parse_args()
    
    env_config, server_configs = MLPerfOptimizerEnv.config_init()
    
    # Override config with CLI arguments if provided
    if hasattr(args, "slurm") and args.slurm is not None:
        slurm = args.slurm == "true"
    else:
        slurm = False
    
    env = MLPerfOptimizerEnv(env_config, server_configs, slurm=slurm, testing=False)
    
    # Run the environment
    asyncio.run(env.run())