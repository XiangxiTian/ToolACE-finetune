"""
Inference script with retrieval module for tool selection.

This script implements:
1. A retrieval module that generates simulated queries for each tool using an LLM
2. Top-k tool retrieval based on similarity between simulated queries and real query
3. Tool selection using the fine-tuned model on the retrieved tools

Usage example:
    python inference.py \
        --model_path ./outputs/qwen3_4b/checkpoint-1272 \
        --base_model_name Qwen/Qwen3-4B-Instruct-2507 \
        --tools_file example_tools.json \
        --query "What's the weather in New York?" \
        --top_k 3 \
        --output_file output.json

For LoRA models, specify both --model_path (LoRA checkpoint) and --base_model_name.
For fully merged models, only specify --model_path.
"""

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from config.base_config import Config
from src.model.model_builder import build_model_and_tokenizer
from src.utils import load_config


class ToolRetrievalModule:
    """Retrieval module that uses LLM-generated simulated queries for tool retrieval."""

    def __init__(
        self,
        query_llm_model: Any,
        query_llm_tokenizer: Any,
        similarity_model: SentenceTransformer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the retrieval module.

        Args:
            query_llm_model: LLM model for generating simulated queries
            query_llm_tokenizer: Tokenizer for the query LLM
            similarity_model: Sentence transformer model for computing similarity
            device: Device to run inference on
        """
        # Move models to device (if not using device_map="auto")
        if not hasattr(query_llm_model, 'hf_device_map') or query_llm_model.hf_device_map is None:
            self.query_llm_model = query_llm_model.to(device)
        else:
            self.query_llm_model = query_llm_model
        self.query_llm_model.eval()
        self.query_llm_tokenizer = query_llm_tokenizer
        self.similarity_model = similarity_model.to(device)
        self.device = device

    def generate_simulated_query(self, tool: Dict[str, Any]) -> str:
        """
        Generate a simulated query for a tool using the LLM.

        Args:
            tool: Tool dictionary with name, description, parameters, etc.

        Returns:
            Simulated query string that would use this tool
        """
        # Format tool information
        tool_name = tool.get("name", "")
        tool_description = tool.get("description", "")
        tool_parameters = tool.get("parameters", {})

        # Create prompt for generating simulated query
        prompt = f"""Given the following tool definition, generate a realistic user query that would require using this tool.

Tool Name: {tool_name}
Tool Description: {tool_description}
Tool Parameters: {json.dumps(tool_parameters, indent=2)}

Generate a concise, natural user query that would lead to using this tool. The query should be similar to what a real user would ask. Only output the query, nothing else."""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates realistic user queries for tools."},
            {"role": "user", "content": prompt}
        ]

        # Format with chat template
        formatted_prompt = self.query_llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate simulated query
        inputs = self.query_llm_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.query_llm_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.query_llm_tokenizer.eos_token_id,
            )

        # Decode the generated text
        generated_text = self.query_llm_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return generated_text

    def retrieve_top_k_tools(
        self,
        tools: List[Dict[str, Any]],
        query: str,
        k: int = 5,
        use_cache: bool = True,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k tools based on similarity between simulated queries and real query.

        Args:
            tools: List of tool dictionaries
            query: Real user query
            k: Number of top tools to retrieve
            use_cache: Whether to cache simulated queries (for efficiency)

        Returns:
            List of (tool, similarity_score) tuples, sorted by similarity (descending)
        """
        if not hasattr(self, "_simulated_queries_cache"):
            self._simulated_queries_cache = {}

        # Generate simulated queries for all tools
        simulated_queries = []
        tool_simulated_pairs = []

        for tool in tools:
            tool_id = tool.get("name", str(id(tool)))

            # Check cache first
            if use_cache and tool_id in self._simulated_queries_cache:
                simulated_query = self._simulated_queries_cache[tool_id]
            else:
                simulated_query = self.generate_simulated_query(tool)
                if use_cache:
                    self._simulated_queries_cache[tool_id] = simulated_query

            simulated_queries.append(simulated_query)
            tool_simulated_pairs.append((tool, simulated_query))

        # Compute embeddings for all simulated queries and the real query
        all_queries = [query] + simulated_queries
        embeddings = self.similarity_model.encode(
            all_queries,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        query_embedding = embeddings[0]
        simulated_embeddings = embeddings[1:]

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            simulated_embeddings,
            dim=1,
        ).cpu().tolist()

        # Pair tools with their similarity scores
        tool_scores = list(zip(tools, similarities))

        # Sort by similarity (descending) and return top-k
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = tool_scores[:k]

        return top_k


class ToolSelectionInference:
    """Tool selection inference using fine-tuned model."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the inference engine.

        Args:
            model: Fine-tuned model for tool selection
            tokenizer: Tokenizer for the model
            device: Device to run inference on
        """
        self.tokenizer = tokenizer
        self.device = device
        # Move model to device (if not using device_map="auto")
        if not hasattr(model, 'hf_device_map') or model.hf_device_map is None:
            self.model = model.to(device)
        else:
            self.model = model
        self.model.eval()

    def format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format tools into a string representation for the prompt.

        Args:
            tools: List of tool dictionaries

        Returns:
            Formatted string representation of tools
        """
        formatted_tools = []
        for tool in tools:
            tool_str = f"Tool: {tool.get('name', 'Unknown')}\n"
            tool_str += f"Description: {tool.get('description', 'No description')}\n"
            
            if "parameters" in tool:
                tool_str += f"Parameters: {json.dumps(tool['parameters'], indent=2)}\n"
            
            formatted_tools.append(tool_str)
        
        return "\n".join(formatted_tools)

    def select_tools(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """
        Select tools using the fine-tuned model.

        Args:
            query: User query
            tools: List of available tools (should be retrieved tools)
            system_prompt: Optional system prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated tool selection response
        """
        # Format tools
        tools_str = self.format_tools_for_prompt(tools)

        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant that selects appropriate tools based on user queries."
            })

        # Add tools and query
        user_content = f"Available tools:\n{tools_str}\n\nUser query: {query}\n\nSelect and use the appropriate tool(s) to answer the query."
        messages.append({"role": "user", "content": user_content})

        # Format with chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return generated_text


def load_finetuned_model(
    model_path: str,
    base_model_name: Optional[str] = None,
    load_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    """
    Load fine-tuned model (with LoRA adapters if applicable).

    Args:
        model_path: Path to fine-tuned model or checkpoint
        base_model_name: Base model name (required if loading LoRA)
        load_in_4bit: Whether to load in 4-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    if base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    if base_model_name:
        # Load base model first, then LoRA adapters
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            pad_token_id=tokenizer.pad_token_id,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None,
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            pad_token_id=tokenizer.pad_token_id,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None,
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Tool selection inference with retrieval")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Base model name (required if loading LoRA adapters)",
    )
    parser.add_argument(
        "--query_llm_model_name",
        type=str,
        default=None,
        help="Model name for generating simulated queries (defaults to base_model_name)",
    )
    parser.add_argument(
        "--similarity_model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for similarity computation",
    )
    parser.add_argument(
        "--tools_file",
        type=str,
        required=True,
        help="Path to JSON file containing list of tools",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="User query (if not provided, will be read from stdin)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top tools to retrieve",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save output (optional)",
    )

    args = parser.parse_args()

    # Load tools
    with open(args.tools_file, "r", encoding="utf-8") as f:
        tools = json.load(f)
    
    if not isinstance(tools, list):
        raise ValueError("Tools file must contain a JSON array of tools")

    print(f"Loaded {len(tools)} tools from {args.tools_file}")

    # Get query
    if args.query:
        query = args.query
    else:
        print("Enter your query (press Ctrl+D when done):")
        query = input().strip()

    if not query:
        raise ValueError("Query cannot be empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load fine-tuned model for tool selection
    print(f"Loading fine-tuned model from {args.model_path}...")
    selection_model, selection_tokenizer = load_finetuned_model(
        args.model_path,
        base_model_name=args.base_model_name,
        load_in_4bit=args.load_in_4bit,
    )
    print("Fine-tuned model loaded successfully")

    # Load query LLM (for generating simulated queries)
    query_llm_name = args.query_llm_model_name or args.base_model_name or args.model_path
    print(f"Loading query LLM from {query_llm_name}...")
    # Create a minimal config for query LLM
    query_config = Config(
        model_name=query_llm_name,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    query_llm_model, query_llm_tokenizer = build_model_and_tokenizer(query_config)
    print("Query LLM loaded successfully")

    # Load similarity model
    print(f"Loading similarity model: {args.similarity_model_name}...")
    similarity_model = SentenceTransformer(args.similarity_model_name)
    print("Similarity model loaded successfully")

    # Initialize retrieval module
    retrieval_module = ToolRetrievalModule(
        query_llm_model=query_llm_model,
        query_llm_tokenizer=query_llm_tokenizer,
        similarity_model=similarity_model,
        device=device,
    )

    # Initialize tool selection
    tool_selection = ToolSelectionInference(
        model=selection_model,
        tokenizer=selection_tokenizer,
        device=device,
    )

    # Step 1: Retrieve top-k tools
    print(f"\nRetrieving top-{args.top_k} tools...")
    top_k_tools_with_scores = retrieval_module.retrieve_top_k_tools(
        tools=tools,
        query=query,
        k=args.top_k,
    )

    top_k_tools = [tool for tool, score in top_k_tools_with_scores]
    print(f"\nRetrieved top-{args.top_k} tools:")
    for i, (tool, score) in enumerate(top_k_tools_with_scores, 1):
        print(f"  {i}. {tool.get('name', 'Unknown')} (similarity: {score:.4f})")

    # Step 2: Tool selection
    print(f"\nPerforming tool selection...")
    response = tool_selection.select_tools(
        query=query,
        tools=top_k_tools,
    )

    # Output results
    print("\n" + "="*80)
    print("QUERY:")
    print(query)
    print("\n" + "="*80)
    print("TOOL SELECTION RESPONSE:")
    print(response)
    print("="*80)

    # Save output if requested
    if args.output_file:
        output = {
            "query": query,
            "retrieved_tools": [
                {
                    "tool": tool,
                    "similarity_score": float(score),
                }
                for tool, score in top_k_tools_with_scores
            ],
            "response": response,
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nOutput saved to {args.output_file}")


if __name__ == "__main__":
    main()
