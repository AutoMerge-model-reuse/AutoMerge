
import subprocess
import tempfile
import yaml
import os
import json
from utils.dynamic_weight import calculate_dynamic_weighted_score

CG_BENCHMARKS = ["humanevalsynthesize-python", "instructioncoder"]

def code_generation_eval_function(model_path: str, benchmark: str, prompt: str, metric_output_path: str) -> float:
    
    command = [
        "accelerate", "launch", "main.py",
        "--model", model_path, 
        "--max_length_generation", "2048",
        "--prompt", prompt, 
        "--tasks", benchmark, 
        "--temperature", "0.8",
        "--do_sample", "True",
        "--n_samples", "10",
        "--batch_size", "5",
        "--allow_code_execution",
        "--save_generations",
        "--save_generations_path", os.path.join(tempfile.gettempdir(), "generations.json"), 
        "--metric_output_path", metric_output_path, 
        "--precision", "bf16"
    ]

    print(f"Executing evaluation command: {' '.join(command)}")


    process = subprocess.run(command, capture_output=True, text=True)

    if process.returncode != 0:
        print(f"Error during evaluation: {process.stderr}")
        return -1000.0 

    score = 0.0
    try:
        with open(metric_output_path, "r", encoding="utf-8") as f:
            eval_results = json.load(f)
            
            if benchmark in eval_results and "pass@1" in eval_results[benchmark]:
                score = eval_results[benchmark]["pass@1"]
            else:
                print(f"Could not find pass@1 score for task {benchmark} in {metric_output_path}")
                score = -1.0 
    except FileNotFoundError:
        print(f"Evaluation result file not found: {metric_output_path}")
        score = -2.0 
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {metric_output_path}")
        score = -3.0 
    finally:

        os.remove(metric_output_path)
        temp_generations_path = os.path.join(tempfile.gettempdir(), "generations.json")
        if os.path.exists(temp_generations_path):
            os.remove(temp_generations_path)
    
    print(f"Evaluation score: {score}")
    return float(score) 

def instruction_eval_function(model_path: str) -> float:
    # benchmark: MMLU-PRO
    pass

def overall_eval_function(model_path: str, benchmarks: list, output_root_path: str) -> float:
    """
    Evaluates the model on multiple benchmarks using code_generation_eval_function
    and computes a weighted overall score.

    Args:
        model_path (str): The path to the merged model.

    Returns:
        float: The weighted overall score across all specified benchmarks.
            Returns a negative value if any individual evaluation fails.
    """
    benchmarks_score = []
    for benchmark in benchmarks:
        if benchmark in CG_BENCHMARKS:
            metric_output_path = os.path.join(output_root_path, f"{benchmark}_eval_result.json")
            score = code_generation_eval_function(model_path, benchmark, prompt="instruction", metric_output_path=metric_output_path)
            benchmarks_score.append(score)
        else:
            metric_output_path = os.path.join(output_root_path, f"{benchmark}_eval_result.json")
            score = instruction_eval_function(model_path)
            benchmarks_score.append(score)
    
    score = calculate_dynamic_weighted_score(benchmarks_score, alpha=1.0)
    
    print(f"Overall evaluation score for model {model_path}: {score}")
    return score


