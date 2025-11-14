import argparse
import json
import os
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def build_messages_from_toolace(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert a ToolACE example to messages format.
    
    Args:
        example: ToolACE dataset example with 'system' and 'conversations'
        
    Returns:
        List of messages in chat format
    """
    messages = []
    
    # Add system message
    messages.append({
        "role": "system",
        "content": example["system"].strip()
    })
    
    for turn in example["conversations"]:
        role = turn["from"].strip().lower()
        
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        elif role == "tool":
            role = "tool"
        
        messages.append({
            "role": role,
            "content": turn["value"].strip()
        })
    
    return messages


def convert_to_sharegpt_format(
    example: Dict[str, Any],
    tokenizer,
    example_id: int
) -> Dict[str, Any]:
    """
    Convert a single ToolACE example to ShareGPT format.
    
    Args:
        example: ToolACE example
        tokenizer: Tokenizer for applying chat template
        example_id: Unique ID for this example
        
    Returns:
        ShareGPT format dict with conversations and metadata
    """
    messages = build_messages_from_toolace(example)
    
    if len(messages) < 2:
        raise ValueError(f"Example {example_id} has too few messages: {len(messages)}")
    
    prompt_messages = messages[:-1]
    expected_completion = messages[-1]["content"]
    
    formatted_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True  # Adds <|im_start|>assistant\n
    )
    
    prompt_tokens = len(tokenizer.encode(formatted_prompt))
    completion_tokens = len(tokenizer.encode(expected_completion))
    
    # Create ShareGPT format
    sharegpt_entry = {
        "conversations": [
            {"value": formatted_prompt},
            {"value": expected_completion}
        ],
        "metadata": {
            "example_id": example_id,
            "num_turns": len(messages),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    
    return sharegpt_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to convert",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random sampling",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/workspaces/gleb_berjoskin/benchmark/toolace_benchmark.json",
        help="Path to output file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Model name",
    )
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    dataset = load_dataset("Team-ACE/ToolACE", split="train")
    dataset = dataset.shuffle(seed=args.seed)
    sampled = dataset.select(range(min(args.num_samples, len(dataset))))
    
    sharegpt_data = []
    
    for idx, example in enumerate(tqdm(sampled, desc="Converting")):
        sharegpt_entry = convert_to_sharegpt_format(example, tokenizer, idx)
        sharegpt_data.append(sharegpt_entry)
    
    cleaned_data = [
        {"conversations": entry["conversations"]}
        for entry in sharegpt_data
    ]
    
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
