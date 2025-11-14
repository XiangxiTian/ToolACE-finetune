import os
import sys
import json
import tiktoken

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from typing import List, Dict

from bfcl_eval.model_handler.utils import formulate_system_prompt
from bfcl_eval.constants.default_prompts import DEFAULT_SYSTEM_PROMPT_FORMAT


def get_gpt4o_tokenizer():
    """Get the GPT-4o tokenizer (o200k_base encoding)."""
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in a text string."""
    return len(tokenizer.encode(text))


def load_toolace_dataset():
    """Load the Toolace dataset from HuggingFace."""
    dataset = load_dataset("Team-ACE/ToolACE", split="train")
    return dataset


def load_bfcl_python_dataset():
    """
    Load the BFCL Python subset dataset from local BFCL package.
    
    According to BFCL's category_mapping.py, the 'python' test category includes:
    - simple_python, irrelevance, parallel, multiple, parallel_multiple
    - live_simple, live_multiple, live_parallel, live_parallel_multiple
    - live_irrelevance, live_relevance
    """

    
    # Python test categories as defined in BFCL
    python_categories = [
        "simple_python",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ]
    
    # Find the bfcl_eval package data directory
    bfcl_data_dir = None
    for path in sys.path:
        potential_dir = os.path.join(path, 'bfcl_eval', 'data')
        if os.path.exists(potential_dir):
            bfcl_data_dir = potential_dir
            break
    
    if bfcl_data_dir is None:
        raise FileNotFoundError("Could not find BFCL data. Make sure bfcl_eval package is installed.")
    
    # Load all Python category JSON files
    all_data = []
    for category in python_categories:
        file_path = os.path.join(bfcl_data_dir, f"BFCL_v4_{category}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
                all_data.extend(data)
            print(f"  Loaded {len(data)} examples from {category}")
        else:
            print(f"  Warning: {file_path} not found, skipping...")
    
    return all_data


def extract_context_from_toolace(example: Dict) -> str:
    """Extract context from a Toolace example."""
    # ToolACE typically has 'query' and 'tools' fields
    # Combine them to form the full context
    context_parts = []
    
    for key in example.keys():
        context_parts.append(str(example[key]))
    
    return "\n".join(context_parts)


def extract_context_from_bfcl(example: Dict) -> str:
    """
    Extract full context from a BFCL example exactly as BFCL constructs it.
    
    This replicates BFCL's system_prompt_pre_processing_chat_model() behavior:
    1. Constructs the default system prompt with embedded function docs
    2. Extracts the user question
    3. Returns: system_prompt + user_question
    """
    # Extract functions
    functions = example.get("function", [])
    
    # Construct the system prompt using BFCL's exact method
    # This uses the default format: ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic
    system_prompt = formulate_system_prompt(
        format_sensitivity_config=DEFAULT_SYSTEM_PROMPT_FORMAT,
        functions=functions
    )
    
    # Extract user question - it's nested as [[{"role": "user", "content": "..."}]]
    user_question = ""
    if "question" in example:
        question_data = example["question"]
        if isinstance(question_data, list) and len(question_data) > 0:
            # Get the first turn (single-turn evaluation)
            first_turn = question_data[0]
            if isinstance(first_turn, list):
                for msg in first_turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        user_question = msg.get("content", "")
                        break
            elif isinstance(first_turn, dict) and first_turn.get("role") == "user":
                user_question = first_turn.get("content", "")
    
    # Combine system prompt and user question (as BFCL does)
    full_context = system_prompt + "\n\n" + user_question
    
    return full_context


def get_context_lengths(dataset, extract_fn, tokenizer) -> List[int]:
    """
    Calculate context lengths for all examples in a dataset.
    
    Args:
        dataset: HuggingFace dataset
        extract_fn: Function to extract context from an example
        tokenizer: Tokenizer to use for counting tokens
    
    Returns:
        List of token counts
    """
    token_counts = []
    for example in dataset:
        context = extract_fn(example)
        tokens = count_tokens(context, tokenizer)
        token_counts.append(tokens)
    return token_counts


def plot_context_length_histogram(token_counts: List[int], dataset_name: str, bins=50):
    """
    Plot histogram of context lengths.
    
    Args:
        token_counts: List of token counts
        dataset_name: Name of the dataset for the title
        bins: Number of bins for the histogram
    """
    plt.figure(figsize=(12, 6))
    plt.hist(token_counts, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Context Length Distribution - {dataset_name}', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_percentiles(token_counts: List[int], percentiles: List[int] = None) -> Dict[str, float]:
    """
    Calculate percentiles for context lengths.
    
    Args:
        token_counts: List of token counts
        percentiles: List of percentiles to calculate (default: [10, 25, 50, 75, 90, 95, 99])
    
    Returns:
        Dictionary with statistics including mean, std, min, max, and percentiles
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 95, 99]
    
    stats = {
        'count': len(token_counts),
        'mean': np.mean(token_counts),
        'std': np.std(token_counts),
        'min': np.min(token_counts),
        'max': np.max(token_counts),
    }
    
    for p in percentiles:
        stats[f'p{p}'] = np.percentile(token_counts, p)
    
    return stats


def print_statistics(stats: Dict[str, float], dataset_name: str):
    """Pretty print the statistics."""
    print(f"\n{'='*50}")
    print(f"Context Length Statistics - {dataset_name}")
    print(f"{'='*50}")
    print(f"Count:    {stats['count']:,}")
    print(f"Mean:     {stats['mean']:.2f}")
    print(f"Std Dev:  {stats['std']:.2f}")
    print(f"Min:      {stats['min']:,}")
    print(f"Max:      {stats['max']:,}")
    print(f"\nPercentiles:")
    for key, value in stats.items():
        if key.startswith('p'):
            percentile = key[1:]
            print(f"  {percentile}th:     {value:,.2f}")
    print(f"{'='*50}\n")


def analyze_dataset(dataset_name: str, load_fn, extract_fn):
    """
    Complete analysis pipeline for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        load_fn: Function to load the dataset
        extract_fn: Function to extract context from examples
    
    Returns:
        Tuple of (token_counts, stats)
    """
    print(f"Loading {dataset_name} dataset...")
    dataset = load_fn()
    print(f"Loaded {len(dataset)} examples")
    
    print(f"Calculating context lengths...")
    tokenizer = get_gpt4o_tokenizer()
    token_counts = get_context_lengths(dataset, extract_fn, tokenizer)
    
    print(f"Calculating statistics...")
    stats = calculate_percentiles(token_counts)
    print_statistics(stats, dataset_name)
    
    print(f"Plotting histogram...")
    plot_context_length_histogram(token_counts, dataset_name)
    
    return token_counts, stats


def analyze_toolace():
    """Analyze ToolACE dataset."""
    return analyze_dataset("ToolACE", load_toolace_dataset, extract_context_from_toolace)


def analyze_bfcl_python():
    """Analyze BFCL Python subset."""
    return analyze_dataset("BFCL Python", load_bfcl_python_dataset, extract_context_from_bfcl)


def compare_distributions(toolace_counts: List[int], bfcl_counts: List[int]):
    """
    Plot both distributions side by side for comparison.
    
    Args:
        toolace_counts: Token counts for ToolACE
        bfcl_counts: Token counts for BFCL Python
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ToolACE
    axes[0].hist(toolace_counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Context Length (tokens)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('ToolACE', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    
    # BFCL
    axes[1].hist(bfcl_counts, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Context Length (tokens)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('BFCL Python', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Overlay comparison
    plt.figure(figsize=(12, 6))
    plt.hist(toolace_counts, bins=50, alpha=0.5, label='ToolACE', color='steelblue', edgecolor='black')
    plt.hist(bfcl_counts, bins=50, alpha=0.5, label='BFCL Python', color='coral', edgecolor='black')
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Context Length Distribution Comparison', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
