
ROUTES = {
        "taskA": "town02",
        "taskB": "town07",
        "base": "town01"
}
TOWNS = {
    "taskA": [1,2],
    "taskB": [1,7],
    "base": [1]
}

import numpy as np

def calculate_dynamic_weighted_score(scenarios_scores, alpha=1.0):
    
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

class GlobalConfig:
    """base architecture configurations"""

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    max_speed = 5
    collision_buffer = [2.5, 1.2]
    model_path = "/mnt/zjy/model_merging/ADmodels/InterFuser/interfuser/output/ckpts/taskA_model_best.pth.tar"
    # model_path = "leaderboard/team_code/interfuser.pth.tar"
    momentum = 0
    skip_frames = 1
    detect_threshold = 0.04

    model = "interfuser_baseline"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def update_model_path(new_model_path, config_class=GlobalConfig, config_file='interfuser_config.py'):
    
    config_class.model_path = new_model_path
    print(f"Updated model_path to: {new_model_path}")
    
    try:
        with open(config_file, 'r') as file:
            lines = file.readlines()

        with open(config_file, 'w') as file:
            for line in lines:
                if line.strip().startswith("model_path"):
                    file.write(f'    model_path = "{new_model_path}"\n')  
                else:
                    file.write(line)

        print(f"Configuration updated successfully in {config_file}")
    
    except Exception as e:
        print(f"Error updating the configuration file: {e}")
        
def replace_routes(sh_path: str, new_route: str):
    
    try:
        with open(sh_path, 'r') as file:
            script_content = file.read()

        updated_content = script_content.replace('town01', new_route)

        with open(sh_path, 'w') as file:
            file.write(updated_content)

        print(f"Successfully replaced 'town01' with '{new_route}' in the script.")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        
def single_task_eval_function(model_path: str, task: str) -> float:
    
    update_model_path(model_path, config_file='/mnt/zjy/model_merging/ADmodels/InterFuser/leaderboard/team_code/interfuser_config.py')
    
    
    if task in ROUTES.keys():
        route = ROUTES[task]
    else:
        raise ValueError(f"Unknown task: {task}. Available tasks are: {list(route.keys())}")

    replace_routes(sh_path='/mnt/zjy/model_merging/ADmodels/InterFuser/leaderboard/scripts/optimize_evaluation.sh', 
                   new_route=route)
    
    
def single_task_val_function(model_path: str, task: str) -> float:
   
    towns = TOWNS[task]
    
    import subprocess
    import os
    import json
    
    conda_env = "interfuser_zjy"
    python_path = "/home/ubuntu/.conda/envs/interfuser_zjy/bin/python"
    
    command = f"export LD_LIBRARY_PATH=/home/ubuntu/.conda/envs/interfuser_zjy/lib/python3.7/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH && \
    {python_path} /mnt/zjy/model_merging/ADmodels/InterFuser/interfuser/val.py --task {task} --ckpt_path {model_path}"
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e.stderr}")
    
    result_path = f"/mnt/zjy/model_merging/mergekit/optimization/scores/validation_result_{task}.json"
    with open(result_path, 'r') as json_file:
        result = json.load(json_file)

    loss = result.get("loss")

    return loss

def val_function(model_path: str, tasks):
    scenarios_scores = []
    for task in tasks:
        print(f"Evaluating model {model_path} on task {task}...")
        score = single_task_val_function(model_path, task)
        scenarios_scores.append(score)
    
    total_score = calculate_dynamic_weighted_score(scenarios_scores, alpha=1.0)
    print(f"Total score for model {model_path} across tasks {tasks}: {total_score}")
    return total_score

if __name__ == "__main__":
    
    model_path = "/mnt/zjy/model_merging/ADmodels/InterFuser/interfuser/output/ckpts/taskA_model_best.pth.tar"
    tasks = ["base", "taskA", "taskB"]
    
    score = val_function(model_path, tasks)
    print(f"Total score for the model {model_path} across tasks {tasks}: {score}")
    
    
    