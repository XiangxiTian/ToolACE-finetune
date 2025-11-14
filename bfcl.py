import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# not the most elegant way to do this, but let's enforce the .env file 
# is loaded before importing bfcl_eval modules to simplify our lives

dotenv_file = Path("./bfcl_dir/.env")
if dotenv_file.exists():
    load_dotenv(dotenv_path=dotenv_file, verbose=True, override=True)
else:
    raise FileNotFoundError(f".env file for bfcl setup not found at {dotenv_file}, adjust the path at line 11 of bfcl_evaluation.py")

from bfcl_eval._llm_response_generation import get_args
from bfcl_eval._llm_response_generation import main as bfcl_generate
from bfcl_eval.eval_checker.eval_runner import main as bfcl_eval_main

# Note: bfcl-eval internal openai server config limits the context length to 4100 tokens,
# this causes openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's
# maximum context length is 4100 tokens. However, your request has 7768 input tokens. Please
# reduce the length of the input messages.", 'type': 'BadRequestError', 'param': None, 'code': 400}} 
# We could patch it, but the whole library is a bit difficult to work with, so we leave it as is.


def get_eval_args():
    """
        This is a direct copy of the argparse code from the bfcl_eval module 
        which we need since eval runner doesn't have the same beautiful neat 
        argparse function as generate runner.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", nargs="+", type=str, help="A list of model names to evaluate"
    )
    parser.add_argument(
        "--test-category",
        nargs="+",
        type=str,
        default="all",
        help="A list of test categories to run the evaluation on",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        type=str,
        help="Path to the folder where the model response files are stored; relative to the `berkeley-function-call-leaderboard` root folder",
    )
    parser.add_argument(
        "--score-dir",
        default=None,
        type=str,
        help="Path to the folder where the evaluation score files will be stored; relative to the `berkeley-function-call-leaderboard` root folder",
    )
    parser.add_argument(
        "--partial-eval",
        default=False,
        action="store_true",
        help="Run evaluation on a partial set of benchmark entries (eg. entries present in the model result files) without raising for missing IDs.",
    )
    parser.add_argument(
        "--served-model-name",
        default="Qwen/Qwen3-4B-Instruct-2507",
        type=str,
        help="The name of the served model",
    )
    args = parser.parse_args()
    return args


def bfcl_generate_wrapper(test_category, model_name, backend="vllm", allow_overwrite=True):
    """
        This is a wrapper around the bfcl_generate function that allows us to 
        launch generation of responses for a given model and test category as 
        a script.
    """
    args_list = [
        "--test-category", test_category,
        "--skip-server-setup",
        "--model", model_name,
        "--backend", backend,
    ]
    
    if allow_overwrite:
        args_list.append("--allow-overwrite")
    
    old_argv = sys.argv
    sys.argv = ["bfcl_evaluation.py"] + args_list
    try:
        args = get_args()
        bfcl_generate(args)
    finally:
        sys.argv = old_argv

def bfcl_evaluate_wrapper(model_name, test_category):
    """
        This is a wrapper around the bfcl_evaluate function that allows us to 
        launch evaluation of responses for a given model and test category as 
        a script.
    """
    args_list = [
        "--model", model_name,
        "--test-category", test_category,
    ]
    
    old_argv = sys.argv
    sys.argv = ["bfcl_evaluation.py"] + args_list
    try:
        args = get_eval_args()
        bfcl_eval_main(args.model, args.test_category, args.result_dir, args.score_dir, args.partial_eval)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    args = get_eval_args()
    model_name = args.served_model_name

    bfcl_generate_wrapper(
        "python", 
        model_name, 
        "vllm",
        allow_overwrite=True
    )
    
    bfcl_evaluate_wrapper(
        model_name,
        "python"
    )

