
import yaml
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from smac import Scenario
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as MFFacade
from smac.facade import HyperparameterOptimizationFacade as HPOFacade
import subprocess
from eval_function.LLM_eval_function import overall_eval_function
import numpy as np

def clean_data(obj):
    if isinstance(obj, np.generic):
        return obj.item()  
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clean_data(value) for key, value in obj.items()}
    return obj

class LLM_merge:
    def __init__(self, yaml_file: str, out_path: str, score_function):
        self.yaml_file = yaml_file
        with open(self.yaml_file, "r", encoding="utf-8") as file:
            self.base_config = yaml.safe_load(file)
        self.score_function = score_function
        self.out_path = out_path
        
    @property
    def configspace(self) -> Configuration:
        cs = ConfigurationSpace()
        cs.add(
            [
                Categorical('merge_method', ['linear', 'ties', 'task_arithmetic', 'dare_linear', 'dare_ties'], default='ties'),
                Float('weight', (0.0, 1.0), default=0.5, log=False),
                Float('density_A', (0.0, 1.0), default=1, log=False),
                Float('density_B', (0.0, 1.0), default=1, log=False)
            ]
        )
        return cs
    
    def train(self, config: Configuration, seed: int = 0) -> float:

        config_dict = dict(config)
        weightA = config_dict["weight"]
        weightB = 1.0 - weightA
        
        models = self.base_config["models"]
        assert len(models) == 2, "LLM_merge requires exactly two models"
        models[0]["parameters"]["weight"] = weightA
        models[1]["parameters"]["weight"] = weightB
        models[0]["parameters"]["density"] = config_dict["density_A"]
        models[1]["parameters"]["density"] = config_dict["density_B"]
        
        temp_config = {
            "models": models,
            "merge_method": config_dict["merge_method"],
            "base_model": self.base_config["base_model"],
            "parameters": self.base_config["parameters"],
            "dtype": self.base_config["dtype"],
        }
        cleaned_config = clean_data(temp_config)

        with open(self.yaml_file, 'w') as file:
            yaml.dump(cleaned_config, file, default_flow_style=False, allow_unicode=True)

        
        command = ['mergekit-pytorch', self.yaml_file, self.out_path]
        try:

            result = subprocess.run(command, check=True, text=True, capture_output=True)

            print("Command output:", result.stdout)
            print("Command error (if any):", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the command: {e}")
            print("Error output:", e.stderr)
            
        score = self.score_function(
            model_path=self.out_path,
            tasks=["base", "taskA", "taskB"]
        )
        
        return score    
        
if __name__ == "__main__":
    merge = LLM_merge(
        yaml_file="/mnt/zjy/model_merging/mergekit/optimization/configs/LLM.yaml",
        out_path="/mnt/zjy/model_merging/mergekit/optimization/ckpts/LLM",
        score_function = overall_eval_function
    )
    scenario = Scenario(
        merge.configspace,
        n_tails = 40,
        n_workers=1
    )
    smac = HPOFacade(
        scenario,
        merge.train,
        initial_design=None,
        overwrite=True
    )
    incumbent = smac.optimize()
    print(f"Best configuration found: {incumbent}")
            