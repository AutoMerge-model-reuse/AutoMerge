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
from datetime import datetime

def clean_data(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # 转换为普通类型（例如 int 或 float）
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clean_data(value) for key, value in obj.items()}
    return obj

def get_gpu_peak_memory(device_index=3):
    """
    获取运行期间 GPU 显存峰值（MB）
    需要 pynvml 库: pip install nvidia-ml-py3
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2  # MB

# ================= GPU 显存监控 =================
class GPUMonitor:
    def __init__(self, device_index=0, interval=1.0, csv_path="gpu_usage.csv", fig_path="gpu_usage.png"):
        self.device_index = device_index
        self.interval = interval
        self.running = False
        self.timestamps = []
        self.memory_usage = []
        self.csv_path = csv_path
        self.fig_path = fig_path
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def _record(self):
        start_time = time.time()
        while self.running:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used_mb = meminfo.used / 1024**2
            elapsed = time.time() - start_time
            self.timestamps.append(elapsed)
            self.memory_usage.append(used_mb)
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
            writer.writerow(["Time(s)", "Memory_Used(MB)"])
            for t, m in zip(self.timestamps, self.memory_usage):
                writer.writerow([t, m])
        print(f"显存采样结果已保存到 {self.csv_path}")

    def save_plot(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.timestamps, self.memory_usage, marker='o', linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("GPU Memory Used (MB)")
        plt.title(f"GPU {self.device_index} Memory Usage Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.fig_path)
        plt.close()
        print(f"显存变化曲线已保存到 {self.fig_path}")

class CCT_Merge:
    def __init__(self, backbone_yaml_file: str, transformer_yaml_file: str, bb_out_path: str, tf_out_path: str, score_function):
        self.backbone_yaml_file = backbone_yaml_file
        self.transformer_yaml_file = transformer_yaml_file
        self.score_function = score_function
        # self.merge_options = merge_options
        self.bb_out_path = bb_out_path
        self.tf_out_path = tf_out_path
        # read the model path from the yaml file, change the hyperparameters for merging
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
        print(f"更新的合并参数已保存到 {self.backbone_yaml_file}")
        
        command = ['mergekit-pytorch', self.backbone_yaml_file, self.bb_out_path]
        try:
            # 使用 subprocess.run 来执行命令，并等待其完成
            result = subprocess.run(command, check=True, text=True, capture_output=True)

            # 打印命令的输出
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
        assert len(models) == 2, "ADs_Merge requires exactly two backbone models"
        
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
        # 将临时配置写入临时文件中
        with open(self.transformer_yaml_file, 'w') as file:
            yaml.dump(cleaned_config, file, default_flow_style=False, allow_unicode=True)
        print(f"更新的合并参数已保存到 {self.transformer_yaml_file}")
        command = ['mergekit-pytorch', self.transformer_yaml_file, self.tf_out_path]
        try:
            # 使用 subprocess.run 来执行命令，并等待其完成
            result = subprocess.run(command, check=True, text=True, capture_output=True)

            # 打印命令的输出
            print("Command output:", result.stdout)
            print("Command error (if any):", result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the command: {e}")
            print("Error output:", e.stderr)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        print(f"模型分块合并完成，输出路径为 {self.bb_out_path} 和 {self.tf_out_path}，开始聚合")
        # 将分别合并的 backbone 和 transformer 整合后进行评估
        aggregation_and_save_weights(
            backbone_weights_path=self.bb_out_path+'/model.safetensors',
            other_weights_path=self.tf_out_path+'/model.safetensors',
            output_path="/mnt/zjy/model_merging/mergekit/optimization/ckpts/CCT/merged/model.pth"
        )
        print("模型聚合完成，开始评估")
        # ========= 仅在 score_function 期间监控 =========
        # 给每次评估的采样文件加上时间戳 & trial 信息，避免覆盖
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"gpu_usage_score_trial{seed}_{ts}.csv"
        fig_path = f"gpu_usage_score_trial{seed}_{ts}.png"

        monitor = GPUMonitor(
            device_index=3,      # 如需改卡号，这里改
            interval=1.0,        # 采样间隔
            csv_path=csv_path,
            fig_path=fig_path,
        )
        monitor.start()
        try:
            score = self.score_function(
                model_path="/mnt/zjy/model_merging/mergekit/optimization/ckpts/CCT/merged/model.pth",
                tasks=["front", "back"]
            )
        finally:
            # 确保异常也能正常停止线程与落盘
            monitor.stop()
            monitor.save_csv()
            monitor.save_plot()
            print(f"[GPU监控] 仅统计 score_function 时段，峰值: {max(monitor.memory_usage):.2f} MB")
            print(f"[GPU监控] 折线图: {fig_path}，CSV: {csv_path}")
        # ========= 监控结束 =========

        return 1 - score
    
if __name__ == "__main__":

    merge = CCT_Merge(
        backbone_yaml_file="/mnt/zjy/model_merging/mergekit/optimization/configs/CCT_configs/backbone.yaml",
        transformer_yaml_file="/mnt/zjy/model_merging/mergekit/optimization/configs/CCT_configs/tf.yaml",
        bb_out_path="/mnt/zjy/model_merging/mergekit/optimization/ckpts/CCT/merged/bb",
        tf_out_path="/mnt/zjy/model_merging/mergekit/optimization/ckpts/CCT/merged/tf",
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
        initial_design=None,  # 可以设置初始设计
        overwrite=True,
    )
    
    incumbent = smac.optimize()
    
    print(f"Best configuration found: {incumbent}")
    