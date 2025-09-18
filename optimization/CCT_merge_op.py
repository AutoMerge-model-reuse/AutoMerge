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
from smac.intensifier.hyperband import Hyperband
import subprocess
import tempfile
import yaml
import os
import numpy as np

from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, add_merge_options

from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving

from eval_function.CCT_eval_function import val_function
from utils.model_split_and_aggregation import aggregation_and_save_weights

import time
import pynvml

import matplotlib.pyplot as plt
from threading import Thread
import csv

def clean_data(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # 转换为普通类型（例如 int 或 float）
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clean_data(value) for key, value in obj.items()}
    return obj

def get_gpu_peak_memory(device_index=3):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2  # MB


class GPUMonitor:
    def __init__(self, device_index=0, interval=1.0, csv_path="gpu_usage.csv", fig_path="gpu_usage.png"):
        self.device_index = device_index
        self.interval = interval
        self.running = False
        self.timestamps = []
        self.memory_usage = []
        self.gpu_utilization = []
        self.gpu_temperature = []
        self.csv_path = csv_path
        self.fig_path = fig_path
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def _record(self):
        start_time = time.time()
        while self.running:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used_mb = meminfo.used / 1024**2
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            temperature = pynvml.nvmlDeviceGetTemperature(self.handle,sensor=0)
            elapsed = time.time() - start_time
            self.timestamps.append(elapsed)
            self.memory_usage.append(used_mb)
            self.gpu_utilization.append(utilization)
            self.gpu_temperature.append(temperature)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = Thread(target=self._record, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def save_csv(self):
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "Memory_Used(MB)", "GPU_Utilization(%)", "GPU_Temperature(C)"])
            for t, m, u, temp in zip(self.timestamps, self.memory_usage, self.gpu_utilization, self.gpu_temperature):
                writer.writerow([t, m, u, temp])
        print(f"GPU result save to {self.csv_path}")



class CCT_Merge:
    def __init__(self, backbone_yaml_file: str, transformer_yaml_file: str, bb_out_path: str, tf_out_path: str, score_function):
        self.backbone_yaml_file = backbone_yaml_file
        self.transformer_yaml_file = transformer_yaml_file
        self.score_function = score_function
        self.bb_out_path = bb_out_path
        self.tf_out_path = tf_out_path
        with open(self.backbone_yaml_file, "r", encoding="utf-8") as file:
            self.backbone_base_config = yaml.safe_load(file)
        with open(self.transformer_yaml_file, "r", encoding="utf-8") as file:
            self.transformer_base_config = yaml.safe_load(file)
            
    @property
    def configspace(self) -> Configuration:
        cs = ConfigurationSpace()
        cs.add(
            [
                Categorical('backbone_merge_method', ['ties','task_arithmetic', 'dare_linear', 'dare_ties'], default='ties'),
                Float('backbone_weight', (0.0, 1.0), default=0.5, log=False),
                Float('backbone_density_A', (0.5, 1.0), default=1, log=False),
                Float('backbone_density_B', (0.5, 1.0), default=1, log=False),
                Categorical('transformer_merge_method', ['ties', 'task_arithmetic', 'dare_linear', 'dare_ties'], default='ties'),
                Float('transformer_weight', (0.0, 1.0), default=0.5, log=False),
                Float('transformer_density_A', (0.5, 1.0), default=1, log=False),
                Float('transformer_density_B', (0.5, 1.0), default=1, log=False)
            ]
        )
        return cs
    
    def train(self, config: Configuration, seed: int = 0) -> float:
        # complete the merge for backbone
        config_dict = dict(config)
        backbone_weightA = config_dict["backbone_weight"]
        backbone_weightB = 1.0 - backbone_weightA
        
        models = self.backbone_base_config["models"]
        assert len(models) == 2, "ADs_Merge requires exactly two backbone models"
        
        models[0]["parameters"]["weight"] = backbone_weightA
        models[1]["parameters"]["weight"] = backbone_weightB
        models[0]["parameters"]["density"] = config_dict["backbone_density_A"]
        models[1]["parameters"]["density"] = config_dict["backbone_density_B"]

        backbone_temp_config = {
            "models": models,
            "merge_method": config_dict["backbone_merge_method"],
            "base_model": self.backbone_base_config["base_model"],
            "parameters": self.backbone_base_config["parameters"],
            "dtype": self.backbone_base_config["dtype"],
        }
        cleaned_config = clean_data(backbone_temp_config)
        
        with open(self.backbone_yaml_file, 'w') as file:
            yaml.dump(cleaned_config, file, default_flow_style=False, allow_unicode=True)
        print(f"update configuration {self.backbone_yaml_file}")
        
        command = ['mergekit-pytorch', self.backbone_yaml_file, self.bb_out_path]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Command output:", result.stdout)
            print("Command error (if any):", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the command: {e}")
            print("Error output:", e.stderr)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        # complete the merge for transformer
        transformer_weightA = config_dict["transformer_weight"]
        transformer_weightB = 1.0 - transformer_weightA
        
        models = self.transformer_base_config["models"]
        assert len(models) == 2, "ADs_Merge requires exactly two transformer models"
        
        models[0]["parameters"]["weight"] = transformer_weightA
        models[1]["parameters"]["weight"] = transformer_weightB
        models[0]["parameters"]["density"] = config_dict["transformer_density_A"]
        models[1]["parameters"]["density"] = config_dict["transformer_density_B"]
        
        transformer_temp_config = {
            "models": models,
            "merge_method": config_dict["transformer_merge_method"],
            "base_model": self.transformer_base_config["base_model"],
            "parameters": self.transformer_base_config["parameters"],
            "dtype": self.transformer_base_config["dtype"],
        }
        cleaned_config = clean_data(transformer_temp_config)
        with open(self.transformer_yaml_file, 'w') as file:
            yaml.dump(cleaned_config, file, default_flow_style=False, allow_unicode=True)
        print(f"update configuration {self.transformer_yaml_file}")
        
        command = ['mergekit-pytorch', self.transformer_yaml_file, self.tf_out_path]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print("Command output:", result.stdout)
            print("Command error (if any):", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the command: {e}")
            print("Error output:", e.stderr)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        aggregation_and_save_weights(
            backbone_weights_path=self.bb_out_path+'/model.safetensors',
            other_weights_path=self.tf_out_path+'/model.safetensors',
            output_path="ckpts/optimization/CCT/merged/model.pth"
        )

        score = self.score_function(
            model_path="ckpts/optimization/CCT/merged/model.pth",
            tasks=["front", "back"]
        )
        return 1 - score

if __name__ == "__main__":

    start_time = time.time()


    monitor = GPUMonitor(device_index=3, interval=1.0)
    monitor.start()
    
    merge = CCT_Merge(
        backbone_yaml_file="optimization/configs/CCT_configs/backbone.yaml",
        transformer_yaml_file="optimization/configs/CCT_configs/tf.yaml",
        bb_out_path="ckpts/optimization/CCT/merged/bb",
        tf_out_path="ckpts/optimization/CCT/merged/tf",
        score_function=val_function
    )
    scenario = Scenario(
        merge.configspace,
        n_trials=200,
        n_workers=1,
    )
    smac = HPOFacade(
        scenario,
        merge.train,
        initial_design=None, 
        overwrite=True,
    )
    
    incumbent = smac.optimize()
    
    print(f"Best configuration found: {incumbent}")
    
    monitor.stop()

    end_time = time.time()
    total_time = end_time - start_time
    
    # save GPU result
    monitor.save_csv()