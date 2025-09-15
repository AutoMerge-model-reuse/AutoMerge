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
    transfuser_weights = {}
    prediction_weights = {}

    
