# 使用 bigcode-eval 库进行代码生成相关能力评估
import subprocess
import tempfile
import yaml
import os
import json
from utils.dynamic_weight import calculate_dynamic_weighted_score

# 定义评估代码生成性能的 benchmark ，定义为常量
CG_BENCHMARKS = ["humanevalsynthesize-python", "instructioncoder"]

def code_generation_eval_function(model_path: str, benchmark: str, prompt: str, metric_output_path: str) -> float:
    # 定义评估结果文件的路径，为了避免冲突，可以使用一个唯一的文件名

    # 构建评估命令
    # 请根据 bigcode-evaluation-harness 的实际路径和您的需求调整以下参数
    command = [
        "accelerate", "launch", "main.py",
        "--model", model_path, # 使用合并后的模型路径
        "--max_length_generation", "2048",
        "--prompt", prompt, # 选择使用的 prompt
        "--tasks", benchmark, # 根据需要修改评估任务
        "--temperature", "0.8",
        "--do_sample", "True",
        "--n_samples", "10",
        "--batch_size", "5",
        "--allow_code_execution",
        "--save_generations",
        # 可以选择不保存 generations，或者保存到临时路径
        "--save_generations_path", os.path.join(tempfile.gettempdir(), "generations.json"), 
        "--metric_output_path", metric_output_path, # 评估结果输出路径
        "--precision", "bf16"
    ]

    print(f"Executing evaluation command: {' '.join(command)}")

    # 执行命令行命令并等待其完成
    process = subprocess.run(command, capture_output=True, text=True)

    # 检查命令执行是否成功
    if process.returncode != 0:
        print(f"Error during evaluation: {process.stderr}")
        # 如果评估失败，返回一个非常小的负值，SMAC 会尝试避免这些配置
        return -1000.0 

    # 从结果文件中读取分数
    score = 0.0
    try:
        with open(metric_output_path, "r", encoding="utf-8") as f:
            eval_results = json.load(f)
            
            # 根据新的 JSON 结构读取 pass@1 分数
            # 评估结果的键是 "multiple-java"，然后是 "pass@1"
            if benchmark in eval_results and "pass@1" in eval_results[benchmark]:
                score = eval_results[benchmark]["pass@1"]
            else:
                print(f"Could not find pass@1 score for task {benchmark} in {metric_output_path}")
                score = -1.0 # 如果找不到分数，返回一个较差的值
    except FileNotFoundError:
        print(f"Evaluation result file not found: {metric_output_path}")
        score = -2.0 # 文件未找到
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {metric_output_path}")
        score = -3.0 # JSON 解析错误
    finally:
        # 清理临时结果文件和生成的 generations 文件
        os.remove(metric_output_path)
        temp_generations_path = os.path.join(tempfile.gettempdir(), "generations.json")
        if os.path.exists(temp_generations_path):
            os.remove(temp_generations_path)
    
    print(f"Evaluation score: {score}")
    return float(score) # SMAC 需要 float 类型的结果

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
            # 根据 benchmark 修改 output 路径
            metric_output_path = os.path.join(output_root_path, f"{benchmark}_eval_result.json")
            score = code_generation_eval_function(model_path, benchmark, prompt="instruction", metric_output_path=metric_output_path)
            benchmarks_score.append(score)
        else:
            metric_output_path = os.path.join(output_root_path, f"{benchmark}_eval_result.json")
            # 评估模型 instruction following 能力
            score = instruction_eval_function(model_path)
            benchmarks_score.append(score)
    
    score = calculate_dynamic_weighted_score(benchmarks_score, alpha=1.0)
    
    print(f"Overall evaluation score for model {model_path}: {score}")
    return score


