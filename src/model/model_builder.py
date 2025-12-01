import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from config.base_config import Config


def _build_quantization_config(config: Config) -> BitsAndBytesConfig | None:
    """
    Build BitsAndBytes quantization config when 4-bit loading is enabled.
    """

    if not config.load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def build_model_and_tokenizer(
    config: Config,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    """
    Load model and tokenizer, and patch the tokenizer for assistant_only_loss.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        
    Returns:
        A tuple of (model, tokenizer) where the tokenizer has been patched
        for use with SFTTrainer's assistant_only_loss=True
    """
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Configure pad token if not set (common for Qwen models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    quantization_config = _build_quantization_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=quantization_config,
        device_map="auto" if quantization_config else None,
    )
    
    # Ensure model config is aligned with tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

