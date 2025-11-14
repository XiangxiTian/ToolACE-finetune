import argparse

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def quantize_fp8_weight_only(
    model_path: str,
    output_dir: str,
    device: str = "cuda",
) -> None:
    """
    Quantize model weights to FP8 while keeping activations in BF16.
    This is weight-only quantization - no calibration needed.
    
    Args:
        model_path: Path to the merged model
        output_dir: Directory to save quantized model
        device: Device to use for quantization
    """
    
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )
    
    oneshot(
        model=model_path,
        dataset=None,
        recipe=recipe,
        output_dir=output_dir,
        trust_remote_code_model=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/qwen3_4b/checkpoint-1272_lora_merged",
        help="Path to merged model (e.g., outputs/qwen3_4b/checkpoint-1272_lora_merged)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen3_4b/checkpoint-1272_lora_merged_fp8",
        help="Directory to save quantized model (e.g., outputs/qwen3_4b/checkpoint-1272_lora_merged_fp8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for quantization (default: cuda)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="FP8_DYNAMIC",
        help="Scheme to use for quantization (default: FP8_DYNAMIC)",
    )
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"{args.model_path}-fp8"
    
    quantize_fp8_weight_only(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
        )

if __name__ == "__main__":
    main()
