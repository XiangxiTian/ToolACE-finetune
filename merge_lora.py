import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_config


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config",
		type=str,
		default="config/qwen3_4b.py",
		help="Path to training config file (e.g., config/qwen3_4b.py)",
	)
	parser.add_argument(
		"--lora_weights_path",
		type=str,
		default="outputs/qwen3_4b/checkpoint-1272",
		help="Path to LoRA adapter weights directory (e.g., outputs/qwen3_14b/)",
	)
	parser.add_argument(
		"--save_merged_path",
		type=str,
		default="outputs/qwen3_4b/checkpoint-1272_lora_merged",
		help="Path to save merged model (e.g., outputs/qwen3_14b_lora_merged)",
	)
	args = parser.parse_args()
	
	print(f"Loading config from: {args.config}")
	config = load_config(args.config)
	base_model_id = config.model_name
	
	print("Loading base model...")
	model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="bfloat16")
	
	print("Loading LoRA weights...")
	model = PeftModel.from_pretrained(model, args.lora_weights_path)
	
	print("Merging LoRA weights into base model...")
	model = model.merge_and_unload()
	
	print("Loading tokenizer...")
	tokenizer = AutoTokenizer.from_pretrained(base_model_id)
	
	print(f"Saving merged model to {args.save_merged_path}...")
	model.save_pretrained(args.save_merged_path)
	tokenizer.save_pretrained(args.save_merged_path)


if __name__ == "__main__":
	main()