from config.base_config import Config, WandbConfig


def get_config() -> Config:
	"""Configuration for Qwen3-4B-Instruct fine-tuning."""
	return Config(
		# Model settings
		model_name="Qwen/Qwen3-4B-Instruct-2507",
		
		# LoRA configuration
		lora_r=64,
		lora_alpha=128,
		lora_dropout=0.05,
		lora_bias="none",
		lora_target_modules=[
			"q_proj",
			"k_proj",
			"v_proj",
			"o_proj",
			"gate_proj",
			"up_proj",
			"down_proj",
		],
		lora_task_type="CAUSAL_LM",
		
		# Dataset settings
		validation_fraction=0.1,
		seed=42,
		
		# Training hyperparameters
		output_dir="./outputs/qwen3_4b",
		num_train_epochs=1.0,
		max_steps=-1, 	
		
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        max_seq_len=4096,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        group_by_length=True,
        dataloader_num_workers=4,
		
        # Logging and evaluation
        logging_steps=10,
        logging_dir="./logs/qwen3_4b",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
       
		
        # Precision
        bf16=True,

                # Quantization
                load_in_4bit=True,
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
			tags=["qwen3-4b"],
			notes="PEFT for Qwen3-4B, ToolACE dataset"
		),
	)

