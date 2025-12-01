import argparse
from typing import Any, Dict

import torch
from transformers import set_seed

import wandb
from src.dataset import get_toolace_datasets
from src.model.model_builder import build_model_and_tokenizer
from src.trainer import build_trainer
from src.utils import initialize_wandb, load_config


def build_text(example: Dict[str, Any], tokenizer, max_seq_length: int, assistant_only_loss: bool = True) -> Dict[str, Any]:
    """Build tokenized inputs from a dataset example with messages format."""

    messages = example.get("messages", [])
    if not messages:
        raise ValueError("Example must contain 'messages' field")

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ).rstrip("\n")

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Create labels - mask non-assistant tokens if assistant_only_loss is True
    if assistant_only_loss:
        labels = mask_non_assistant_tokens(messages, tokenizer, input_ids, max_seq_length)
    else:
        labels = input_ids.copy()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def mask_non_assistant_tokens(
    messages: list[Dict[str, str]],
    tokenizer,
    full_input_ids: list[int],
    max_seq_length: int,
) -> list[int]:
    """
    Mask non-assistant tokens in labels by setting them to -100.
    Only assistant message tokens will contribute to the loss.
    
    This works by tokenizing the conversation incrementally and identifying
    which tokens belong to assistant messages.
    """
    labels = [-100] * len(full_input_ids)
    
    # Tokenize the conversation incrementally to identify assistant tokens
    for i in range(len(messages)):
        message = messages[i]
        
        # Tokenize messages up to (but not including) current message
        if i > 0:
            prev_messages = messages[:i]
            text_prev = tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=False,
            ).rstrip("\n")
            tokenized_prev = tokenizer(
                text_prev,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
            prev_tokens = tokenized_prev["input_ids"]
            prev_len = len(prev_tokens)
        else:
            prev_len = 0
        
        # Tokenize messages up to and including current message
        messages_up_to_here = messages[:i+1]
        text_up_to_here = tokenizer.apply_chat_template(
            messages_up_to_here,
            tokenize=False,
            add_generation_prompt=False,
        ).rstrip("\n")
        tokenized_up_to_here = tokenizer(
            text_up_to_here,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokens_up_to_here = tokenized_up_to_here["input_ids"]
        current_len = len(tokens_up_to_here)
        
        # If this is an assistant message, mark the new tokens as assistant tokens
        if message.get("role") == "assistant":
            # The tokens from prev_len to current_len (in the incremental sequence)
            # correspond to this assistant message. Map them to the full sequence.
            # We need to find where these tokens appear in full_input_ids.
            
            # Get the assistant message tokens from the incremental sequence
            assistant_tokens_incremental = tokens_up_to_here[prev_len:current_len]
            
            # Find where this sequence appears in full_input_ids
            # Start searching from where we expect it (after previous messages)
            search_start = min(prev_len, len(full_input_ids))
            
            # Try to find the assistant tokens in the full sequence
            for start_idx in range(search_start, len(full_input_ids) - len(assistant_tokens_incremental) + 1):
                if full_input_ids[start_idx:start_idx + len(assistant_tokens_incremental)] == assistant_tokens_incremental:
                    # Mark these tokens as assistant tokens
                    for j in range(len(assistant_tokens_incremental)):
                        if start_idx + j < len(labels):
                            labels[start_idx + j] = full_input_ids[start_idx + j]
                    break
    
    return labels


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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    report_to, run_name = initialize_wandb(config)

    model, tokenizer = build_model_and_tokenizer(config)

    train_ds, val_ds = get_toolace_datasets(
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    def map_build_text(example):
        return build_text(example, tokenizer, config.max_seq_len, config.assistant_only_loss)

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
