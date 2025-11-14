from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WandbConfig:
	"""Weights & Biases monitoring configuration."""
	enabled: bool = True
	project: str = "llm-function-calling"
	run_name: Optional[str] = None 
	entity: Optional[str] = None 
	tags: list[str] = field(default_factory=list)
	notes: Optional[str] = None


@dataclass
class Config:
	"""Unified configuration for model training."""
	
	# Model settings
	model_name: str = None
	
	# LoRA configuration
	lora_r: int = 8
	lora_alpha: int = 16
	lora_dropout: float = 0.05
	lora_bias: str = "none"
	lora_target_modules: list[str] = field(default_factory=lambda: [
		"q_proj", "k_proj", "v_proj", "o_proj",
		"gate_proj", "up_proj", "down_proj"
	])
	lora_task_type: str = "CAUSAL_LM"
	
	# Dataset settings
	validation_fraction: float = 0.1
	seed: int = 42
	
	# Training hyperparameters
	output_dir: str = "./outputs"
	num_train_epochs: float = 1.0
	max_steps: int = -1 
	per_device_train_batch_size: int = 3
	per_device_eval_batch_size: int = 1
	learning_rate: float = 2e-5
	max_seq_len: int = 2048
	gradient_accumulation_steps: int = 4
	weight_decay: float = 0.01
	
	# Logging and evaluation
	logging_steps: int = 20
	save_strategy: str = "epoch"
	save_steps: int = 500 
	save_total_limit: Optional[int] = None
	eval_strategy: str = "epoch"
	eval_steps: int = 500
	
	# Precision
	bf16: bool = False
	
	# Other training settings
	warmup_ratio: float = 0.03
	assistant_only_loss: bool = True
	logging_dir: str = "./logs"
	
	# Wandb monitoring
	wandb: WandbConfig = field(default_factory=WandbConfig)

