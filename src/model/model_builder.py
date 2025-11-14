from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def build_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    """
    Load model and tokenizer, and patch the tokenizer for assistant_only_loss.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        
    Returns:
        A tuple of (model, tokenizer) where the tokenizer has been patched
        for use with SFTTrainer's assistant_only_loss=True
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure pad token if not set (common for Qwen models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Ensure model config is aligned with tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

