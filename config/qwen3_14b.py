from config.base_config import Config, WandbConfig


def get_config() -> Config:
	"""Configuration for Qwen3-14B fine-tuning."""
	return Config(
		# Model settings
		model_name="Qwen/Qwen3-14B",	
		# LoRA configuration
		lora_r=16,
		lora_alpha=32,
		lora_dropout=0.1,
		lora_bias="none",
		lora_target_modules=[
			"q_proj",
			"k_proj",
			"v_proj",
		],
		lora_task_type="CAUSAL_LM",
		
		# Dataset settings
		validation_fraction=0.1,
		seed=42,
		
		# Training hyperparameters
		output_dir="./outputs/qwen3_14b",
		num_train_epochs=3.0,
		max_steps=-1,
		per_device_train_batch_size=1,
		per_device_eval_batch_size=1,
		learning_rate=5e-6,
		max_seq_len=8192, # more than any sequence in training dataset
		gradient_accumulation_steps=8,
		weight_decay=0.01,
		
        # Logging and evaluation
        logging_steps=10,
        logging_dir="./logs/qwen3_14b",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
		
        # Precision
        bf16=True,

                # Quantization
                load_in_4bit=False,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
		
		# Other training settings
		warmup_ratio=0.05,
		assistant_only_loss=True,
		
		# Wandb monitoring
		wandb=WandbConfig(
			enabled=True,
			project="ToolACE-finetune",
			tags=["qwen3-14b"],
			notes="PEFT for Qwen3-14B, ToolACE dataset"
		),
	)

