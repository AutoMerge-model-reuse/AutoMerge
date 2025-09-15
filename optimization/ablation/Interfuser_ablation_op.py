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

from eval_function.ADs_eval_function import val_function
from utils.model_split_and_aggregation import aggregation_and_save_weights

# 消融实验部分和完整方法不同的地方在于参数空间的定义以及是否进行预训练权重的切分以及聚合

def clean_data(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # 转换为普通类型（例如 int 或 float）
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clean_data(value) for key, value in obj.items()}
    return obj

class ADS_Merge:
    def __init__(self, model_yaml_file: str, model_out_path: str, score_function):
        self.model_yaml_file = model_yaml_file
        self.score_function = score_function
        # self.merge_options = merge_options
        self.model_out_path = model_out_path
        # 从源 yaml 中读入模型名称，后续只对可变超参数进行更改用于合并
        
        with open(self.model_yaml_file, "r", encoding="utf-8") as file:
            self.model_base_config = yaml.safe_load(file)
        