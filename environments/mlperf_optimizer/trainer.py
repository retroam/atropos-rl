"""
MLPerf Optimizer Trainer

This script handles the training loop for the MLPerf optimization environment.
It loads a base model, sets up optimization, and trains the model using GRPO.
"""

import argparse
import json
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import requests
import torch
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variable to keep track of the vLLM process
vllm_process = None

class TrainingConfig(BaseModel):
    """Configuration for MLPerf optimizer training."""
    model_name: str = Field(..., description="Name of the base model to train")
    lr: float = Field(1e-5, description="Learning rate for optimizer")
    training_steps: int = Field(500, description="Number of training steps")
    batch_size: int = Field(4, description="Batch size for training")
    seq_len: int = Field(2048, description="Maximum sequence length")
    device: str = Field("cuda", description="Device to run training on ('cuda', 'cpu')")
    save_path: str = Field("./checkpoints", description="Path to save model checkpoints")
    lora: bool = Field(True, description="Whether to use LoRA for training")
    lora_r: int = Field(8, description="LoRA r dimension")
    lora_alpha: float = Field(16, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.05, description="LoRA dropout")
    advantage_clip: float = Field(10.0, description="Advantage clipping parameter for GRPO")
    kl_coef: float = Field(0.1, description="KL coefficient for GRPO")
    save_interval: int = Field(50, description="Save model every N steps")
    vllm_port: int = Field(9001, description="Port for vLLM server")
    vllm_restart_interval: int = Field(100, description="Restart vLLM server every N steps")
    use_wandb: bool = Field(True, description="Whether to use Weights & Biases for logging")
    wandb_project: Optional[str] = Field("", description="Wandb project name")
    wandb_group: Optional[str] = Field("", description="Wandb group name")
    warmup_steps: int = Field(100, description="Number of warmup steps for LR scheduler")
    gradient_accumulation_steps: int = Field(8, description="Number of gradient accumulation steps")
    log_interval: int = Field(10, description="Log every N steps")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def register_trainer(config: TrainingConfig):
    """Register trainer with orchestration server."""
    requests.post(
        "http://localhost:8000/register",
        json={
            "wandb_group": config.wandb_group or "",
            "wandb_project": config.wandb_project or "",
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "num_steps": config.training_steps,
        },
        timeout=10,
    )

def get_batch():
    """Get a batch from the orchestration server."""
    try:
        r = requests.post("http://localhost:8000/batch", timeout=60)
        r.raise_for_status()
        try:
            data = r.json()
            return data
        except json.JSONDecodeError:
            print("Failed to decode JSON from /batch response")
            print(f"Response text: {r.text[:1000]}")
            return {"batch": None}
    except (requests.RequestException, ConnectionError) as e:
        print(f"Error fetching batch: {e}")
        return {"batch": None}

def pad_data_to_good_offset(data, batch_size):
    """Process and pad batch data to proper dimensions."""
    tokenizations = data["batch"]["tokenizations"]
    advantages = data["batch"]["advantages"]

    tokenization_result = []
    all_advantages = []
    all_masks = []
    
    # Process all tokenizations in the batch
    for idx, tk in enumerate(tokenizations):
        input_ids = torch.tensor(tk["input_ids"])
        attention_mask = torch.tensor(tk["attention_mask"])

        # Convert advantages to tensor
        advantage = torch.tensor(advantages[idx])
        
        tokenization_result.append(input_ids)
        all_advantages.append(advantage)
        all_masks.append(attention_mask)
    
    return tokenization_result, all_advantages, all_masks

def get_data(
    batch_size: int, seq_len: int
) -> List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
    """Get data from orchestration server."""
    batches = []
    while True:
        data = get_batch()
        if data["batch"] is not None:
            batches.append(pad_data_to_good_offset(data, batch_size))
        elif len(batches) > 0:
            return batches
        else:
            time.sleep(1)

def train(config: TrainingConfig):
    """Main training function."""
    global vllm_process

    # Initialize Weights & Biases if enabled
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project or "mlperf_optimizer",
                group=config.wandb_group or None,
                config=config.dict(),
            )
        except ImportError:
            print("Weights & Biases not installed. Skipping wandb initialization.")
            config.use_wandb = False
    
    # Create save directory if it doesn't exist
    os.makedirs(config.save_path, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model: {config.model_name}")
    if config.lora:
        try:
            from peft import (LoraConfig, TaskType, get_peft_model,
                              prepare_model_for_kbit_training)
            
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
                device_map="auto" if config.device == "cuda" else None,
            )
            
            # Prepare model for LoRA training
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            
            # Setup LoRA configuration
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
        except ImportError:
            print("PEFT not installed. Falling back to full model training.")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map="auto" if config.device == "cuda" else None,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto" if config.device == "cuda" else None,
        )
    
    # Important for gradient accumulation to work properly
    model.config.use_cache = False
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    # Setup learning rate scheduler with warmup
    from transformers import get_cosine_schedule_with_warmup
    total_steps = config.training_steps
    num_warmup_steps = min(config.warmup_steps, total_steps // 10)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Register trainer with orchestration server
    register_trainer(config)
    
    # Initialize vLLM server
    vllm_command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.model_name,
        "--port", str(config.vllm_port),
        "--tensor-parallel-size", "1",
    ]
    print(f"Starting vLLM server: {' '.join(vllm_command)}")
    vllm_process = subprocess.Popen(vllm_command)
    
    # Wait for vLLM server to start
    time.sleep(20)
    
    # Training loop
    step = 0
    log_loss = 0
    log_kl_loss = 0
    log_pg_loss = 0
    accumulated_steps = 0
    
    while step < config.training_steps:
        # Get batches from orchestration server
        print(f"Getting data for step {step}")
        batches = get_data(config.batch_size, config.seq_len)
        
        for batch_idx, (token_ids, advantages, attention_masks) in enumerate(batches):
            if step >= config.training_steps:
                break
            
            # Process batch
            input_ids_list = []
            attention_mask_list = []
            advantage_list = []
            
            # Prepare batch data
            for ids, mask, adv in zip(token_ids, attention_masks, advantages):
                input_ids_list.append(ids)
                attention_mask_list.append(mask)
                advantage_list.append(adv)
            
            if not input_ids_list:  # Skip empty batches
                continue
            
            # Pad sequences to maximum length in batch
            max_length = max(len(ids) for ids in input_ids_list)
            padded_input_ids = torch.stack([
                torch.cat([
                    torch.full((max_length - len(ids),), tokenizer.pad_token_id, dtype=ids.dtype),
                    ids
                ]) if len(ids) < max_length else ids
                for ids in input_ids_list
            ])
            
            padded_attention_masks = torch.stack([
                torch.cat([
                    torch.zeros(max_length - len(mask), dtype=mask.dtype),
                    mask
                ]) if len(mask) < max_length else mask
                for mask in attention_mask_list
            ])
            
            # Pad advantages
            padded_advantages = torch.stack([
                torch.cat([
                    torch.zeros(max_length - len(adv), dtype=adv.dtype),
                    adv
                ]) if len(adv) < max_length else adv
                for adv in advantage_list
            ])
            
            # Move tensors to device
            padded_input_ids = padded_input_ids.to(config.device)
            padded_attention_masks = padded_attention_masks.to(config.device)
            padded_advantages = padded_advantages.to(config.device)
            
            # Forward pass with gradient accumulation
            outputs = model(
                input_ids=padded_input_ids,
                attention_mask=padded_attention_masks,
                use_cache=False,
                return_dict=True,
            )
            
            logits = outputs.logits
            
            # GRPO loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = padded_input_ids[..., 1:].contiguous()
            shift_attention_mask = padded_attention_masks[..., 1:].contiguous()
            shift_advantages = padded_advantages[..., 1:].contiguous()
            
            # Get predicted token probabilities
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Calculate policy gradient loss (focused on tokens with non-zero advantages)
            mask = shift_attention_mask.bool() & (shift_advantages != 0)
            
            # Clip advantages for stability
            clipped_advantages = torch.clamp(
                shift_advantages, -config.advantage_clip, config.advantage_clip
            )
            
            # Calculate policy gradient loss
            pg_loss = -(token_log_probs * clipped_advantages * mask).sum() / (mask.sum() + 1e-8)
            
            # KL penalty to prevent diverging too far from the base model
            kl_loss = 0.0
            if config.kl_coef > 0:
                with torch.no_grad():
                    # We would need the reference model here in a full implementation
                    # For the hackathon, we'll just use a simplified KL penalty
                    kl_loss = -token_log_probs.mean() * config.kl_coef
            
            # Combine losses
            loss = pg_loss + kl_loss
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update log values
            log_loss += loss.item() * config.gradient_accumulation_steps
            log_pg_loss += pg_loss.item()
            log_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            
            accumulated_steps += 1
            
            # Perform optimizer step after accumulation
            if accumulated_steps % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                if step % config.log_interval == 0:
                    avg_loss = log_loss / min(config.log_interval, step + 1)
                    avg_pg_loss = log_pg_loss / min(config.log_interval, step + 1)
                    avg_kl_loss = log_kl_loss / min(config.log_interval, step + 1)
                    
                    print(f"Step {step} - Loss: {avg_loss:.6f}, PG Loss: {avg_pg_loss:.6f}, KL Loss: {avg_kl_loss:.6f}, LR: {lr_scheduler.get_last_lr()[0]:.8f}")
                    
                    if config.use_wandb:
                        wandb.log({
                            "loss": avg_loss,
                            "pg_loss": avg_pg_loss,
                            "kl_loss": avg_kl_loss,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                        }, step=step)
                    
                    log_loss = 0
                    log_pg_loss = 0
                    log_kl_loss = 0
                
                # Save checkpoint
                if (step + 1) % config.save_interval == 0 or step == config.training_steps - 1:
                    checkpoint_path = os.path.join(config.save_path, f"step_{step+1}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    print(f"Saving checkpoint to {checkpoint_path}")
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                
                # Restart vLLM server with updated model
                if (step + 1) % config.vllm_restart_interval == 0 or step == config.training_steps - 1:
                    checkpoint_path = os.path.join(config.save_path, f"step_{step+1}")
                    
                    # Terminate existing vLLM process
                    if vllm_process:
                        print("Terminating vLLM server")
                        vllm_process.terminate()
                        vllm_process.wait()
                    
                    # Launch new vLLM server with updated model
                    updated_vllm_command = [
                        "python", "-m", "vllm.entrypoints.openai.api_server",
                        "--model", checkpoint_path,
                        "--port", str(config.vllm_port),
                        "--tensor-parallel-size", "1",
                    ]
                    print(f"Starting new vLLM server with updated model: {' '.join(updated_vllm_command)}")
                    vllm_process = subprocess.Popen(updated_vllm_command)
                    
                    # Wait for vLLM server to start
                    time.sleep(20)
                
                step += 1
                if step >= config.training_steps:
                    break
    
    # Save final model
    final_path = os.path.join(config.save_path, "final")
    os.makedirs(final_path, exist_ok=True)
    
    print(f"Saving final model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Clean up vLLM process
    if vllm_process:
        vllm_process.terminate()
        vllm_process.wait()
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLPerf Optimizer Trainer")
    parser.add_argument("--model_name", type=str, default="NousResearch/DeepHermes-3-Llama-3-3B-Preview", 
                        help="Name of the base model to train")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--training_steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on ('cuda', 'cpu')")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="Path to save checkpoints")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for training")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="", help="WandB project name")
    parser.add_argument("--wandb_group", type=str, default="", help="WandB group name")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model_name,
        lr=args.lr,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        save_path=args.save_path,
        lora=args.use_lora,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
    )
    
    train(config)