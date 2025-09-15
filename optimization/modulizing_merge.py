import subprocess
import tempfile
import yaml
import os
import numpy as np
from utils.model_split_and_aggregation import aggregation_and_save_weights

''' 
根据参数调优得到的参数组合进行分块合并
读取 backbone 和 transformer 的基础配置文件，根据 config_path 中的参数进行修改，
修改配置文件后分块合并，然后聚合
'''

def clean_data(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # 转换为普通类型（例如 int 或 float）
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clean_data(value) for key, value in obj.items()}
    return obj

def modulization_and_merge(config_path: str, backbone_yaml_file: str, transformer_yaml_file: str, bb_out_path: str, tf_out_path: str, output_path: str):
    with open(backbone_yaml_file, "r", encoding="utf-8") as file:
        backbone_base_config = yaml.safe_load(file)
    with open(transformer_yaml_file, "r", encoding="utf-8") as file:
        transformer_base_config = yaml.safe_load(file)
    with open(config_path, 'r') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    backbone_weightA = config_dict["backbone_weight"]
    backbone_weightB = 1.0 - backbone_weightA
    backbone_densityA = config_dict["backbone_density_A"]
    backbone_densityB = config_dict["backbone_density_B"]
    
    models = backbone_base_config["models"]
    assert len(models) == 2, "ADs_Merge requires exactly two backbone models"
    
    models[0]["parameters"]["weight"] = backbone_weightA
    models[1]["parameters"]["weight"] = backbone_weightB
    models[0]["parameters"]["density"] = backbone_densityA
    models[1]["parameters"]["density"] = backbone_densityB

    backbone_temp_config = {
        "models": models,
        "merge_method": config_dict["backbone_merge_method"],
        "base_model": backbone_base_config["base_model"],
        "parameters": backbone_base_config["parameters"],
        "dtype": backbone_base_config["dtype"],
    }
    cleaned_config = clean_data(backbone_temp_config)
    # 将临时配置写入临时文件中
    with open(backbone_yaml_file, 'w') as file:
        yaml.dump(cleaned_config, file, default_flow_style=False, allow_unicode=True)
    print(f"更新的合并参数已保存到 {backbone_yaml_file}")
        
    command = ['mergekit-pytorch', backbone_yaml_file, bb_out_path]
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

    transformer_weightA = config_dict["transformer_weight"]
    transformer_weightB = 1.0 - transformer_weightA
    transformer_densityA = config_dict["transformer_density_A"]
    transformer_densityB = config_dict["transformer_density_B"]
    
    models = transformer_base_config["models"]
    assert len(models) == 2, "ADs_Merge requires exactly two backbone models"
    
    models[0]["parameters"]["weight"] = transformer_weightA
    models[1]["parameters"]["weight"] = transformer_weightB
    models[0]["parameters"]["density"] = transformer_densityA
    models[1]["parameters"]["density"] = transformer_densityB
    
    transformer_temp_config = {
        "models": models,
        "merge_method": config_dict["transformer_merge_method"],
        "base_model": transformer_base_config["base_model"],
        "parameters": transformer_base_config["parameters"],
        "dtype": transformer_base_config["dtype"],
    }
    cleaned_config = clean_data(transformer_temp_config)
    # 将临时配置写入临时文件中
    with open(transformer_yaml_file, 'w') as file:
        yaml.dump(cleaned_config, file, default_flow_style=False, allow_unicode=True)
    print(f"更新的合并参数已保存到 {transformer_yaml_file}")
        
    command = ['mergekit-pytorch', transformer_yaml_file, tf_out_path]
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
    
    print(f"模型分块合并完成，输出路径为 {bb_out_path} 和 {tf_out_path}，开始聚合")
    # 将分别合并的 backbone 和 transformer 整合后进行评估
    aggregation_and_save_weights(
        backbone_weights_path=bb_out_path+'/model.safetensors',
        other_weights_path=tf_out_path+'/model.safetensors',
        output_path=output_path
    )
    print("模型聚合完成")
    
if __name__ == "__main__":
    config_path = "/mnt/zjy/model_merging/mergekit/optimization/configs/CCT_configs/optimization_result.yaml"
    backbone_yaml_file = "/mnt/zjy/model_merging/mergekit/optimization/configs/CCT_configs/backbone.yaml"
    transformer_yaml_file = "/mnt/zjy/model_merging/mergekit/optimization/configs/CCT_configs/tf.yaml"
    bb_out_path = "/mnt/zjy/model_merging/mergekit/optimization/ckpts/CCT/merged/bb"
    tf_out_path = "/mnt/zjy/model_merging/mergekit/optimization/ckpts/CCT/merged/tf"
    output_path = "/mnt/zjy/model_merging/mergekit/optimization/ckpts/CCT/merged/opt_result.pth"

    modulization_and_merge(config_path, backbone_yaml_file, transformer_yaml_file, bb_out_path, tf_out_path, output_path)