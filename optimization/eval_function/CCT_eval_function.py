import numpy as np
import subprocess
import os
import json

def calculate_dynamic_weighted_score(scenarios_scores, alpha=1.0):
    """
    计算动态加权总分，根据每个任务的评分表现动态调整权重。

    参数:
    scenarios_scores: 一个列表或数组，包含每个场景的评分。
    alpha: 调整因子，控制权重调整的灵敏度。

    return:
    总分（float类型）
    """
    min_score = np.min(scenarios_scores)
    max_score = np.max(scenarios_scores)

    # normalized_scores = [(score - min_score) / (max_score - min_score) for score in scenarios_scores]

    # mean_score = np.mean(normalized_scores)


    mean_score = np.mean(scenarios_scores)

    differences = [abs(score - mean_score) for score in scenarios_scores]

    max_diff = max(differences)
    weights = [(1 / len(scenarios_scores)) * (1 + alpha * diff / max_diff) for diff in differences]

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    total_score = sum(w * s for w, s in zip(normalized_weights, scenarios_scores))

    return total_score

def single_task_val_function(model_path: str, task: str) -> float:
    """
    执行单任务验证函数
    :param model_path: 模型路径
    :param task: 任务名称，i.e., 'front', 'back'    
    :return: 验证分数
    """
    python_path = "/home/ubuntu/.conda/envs/cct/bin/python"
    
    # create the eval command along with the model_path annd task
    command = f"export LD_LIBRARY_PATH=/home/ubuntu/.conda/envs/cct/lib/python3.8/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH && \
    {python_path} /mnt/zjy/model_merging/ADmodels/Compact-Transformers/eval.py --model_path {model_path} --dataset {task}"
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e.stderr}")
        
    result_path = f"/mnt/zjy/model_merging/mergekit/optimization/scores/CCT_scores/eval_{task}_result.json"
    
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)   # 读取成 Python 字典

    top5 = data["top5_accuracy"]
    
    return top5

def val_function(model_path: str, tasks):
    scores = []
    for task in tasks:
        print(f"Evaluating model {model_path} on task {task}...")
        score = single_task_val_function(model_path, task)
        scores.append(score)
        
    totol_score = calculate_dynamic_weighted_score(scores, alpha=1.0)
    print(f"Total score for model {model_path} is {totol_score}")
    return totol_score

if __name__ == "__main__":
    val_function(
        model_path="/mnt/zjy/model_merging/mergekit/optimization/ckpts/merged/ADs.pth",
        tasks=["front", "back"]
    )