import torch
from safetensors.torch import load_file

def aggregation_and_save_weights(backbone_weights_path, other_weights_path, output_path):
    # Load the separated weights using safetensors
    backbone_weights = load_file(backbone_weights_path)
    other_weights = load_file(other_weights_path)

    # Merge the weights
    merged_weights = {}
    merged_weights.update(backbone_weights)
    merged_weights.update(other_weights)

    # Save the merged weights in .pth format
    torch.save(merged_weights, output_path)

    print(f"Merged weights have been saved to {output_path}!")
    
def separate_and_save_weights(model_weights_path, output_dir):
    # Load the model weights (assuming the weights are in a .pth file)
    model_weights = torch.load(model_weights_path)
    
    # Initialize dictionaries to hold the separated weights
    backbone_weights = {}
    other_weights = {}

    # Iterate through model weights to categorize the keys
    for key in model_weights.keys():
        if any(part in key for part in ['rgb_backbone', 'lidar_backbone', 'rgb_patch_embed', 'lidar_patch_embed']):
            backbone_weights[key] = model_weights[key]
        else:
            other_weights[key] = model_weights[key]

    # Save the separated weights
    torch.save(backbone_weights, f'{output_dir}/backbone_weights.pth')
    torch.save(other_weights, f'{output_dir}/other_weights.pth')

    print("Weights have been separated and saved!")
