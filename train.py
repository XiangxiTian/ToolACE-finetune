import argparse
from typing import Any, Dict

import torch
from transformers import set_seed

import wandb
from merge_lora import load_config
from src.dataset import get_toolace_datasets
from src.model.model_builder import build_model_and_tokenizer
from src.trainer import build_trainer
from src.utils import initialize_wandb, load_config


def build_text(example: Dict[str, Any], tokenizer, max_seq_length: int) -> Dict[str, Any]:
	"""
	Build text and input_ids from a dataset example with messages format.
	
	Args:
		example: Dataset example with 'messages' field containing conversation
		tokenizer: The tokenizer to use for encoding
		max_seq_length: Maximum sequence length for truncation
		
	Returns:
		Dictionary with 'text' and 'input_ids' fields
	"""
	# Extract messages from the example
	messages = example.get('messages', [])
	
	if not messages:
		raise ValueError("Example must contain 'messages' field")
	
	# Apply chat template without tokenization first
	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=False,
	)
	text = text.rstrip('\n')
	
	return {"text": text}

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config",
		type=str,
		default="config/qwen3_4b.py",
		help="Path to config file (e.g., config/qwen3_14b.py)",
	)
	parser.add_argument(
		"--resume",
		type=bool,
		default=False,  
		help="Resume training from the latest checkpoint in output_dir",
	)
	args = parser.parse_args()
	
	config = load_config(args.config)
	print(f"Loaded config from: {args.config}")
	print(f"Model: {config.model_name}")
	
	set_seed(config.seed)  
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	report_to, run_name = initialize_wandb(config)
	
	model, tokenizer = build_model_and_tokenizer(config.model_name)

	train_ds, val_ds = get_toolace_datasets(
		validation_fraction=config.validation_fraction,
		seed=config.seed,
	)
	
	def map_build_text(example):
		return build_text(example, tokenizer, config.max_seq_len)
	
	train_dataset = train_ds.map(map_build_text, remove_columns=train_ds.column_names)
	eval_dataset = val_ds.map(map_build_text, remove_columns=val_ds.column_names)
	
	trainer = build_trainer(
		config=config,
		model=model,
		tokenizer=tokenizer,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		report_to=report_to,
		run_name=run_name,
	)

	# Only resume if explicitly requested, otherwise train from scratch
	if args.resume:
		print("Resuming from latest checkpoint...")
		trainer.train(resume_from_checkpoint=True)
	else:
		print("Training from scratch (ignoring any existing checkpoints)...")
		trainer.train(resume_from_checkpoint=False)
  
	trainer.model.save_pretrained(config.output_dir)
	tokenizer.save_pretrained(config.output_dir)
	print(f"LoRA adapter weights saved to: {config.output_dir}")
	
	if config.wandb.enabled:
		wandb.finish()

if __name__ == "__main__":
	main()


