import torch

def separate_and_save_weights(model_weights_path, output_dir):
    # Load the model weights (assuming the weights are in a .pth file)
    model_weights = torch.load(model_weights_path)
    
    # Initialize dictionaries to hold the separated weights
    backbone_weights = {}
    other_weights = {}

    # Iterate through model weights to categorize the keys
    for key in model_weights.keys():
        if any(part in key for part in ['tokenizer']):
            backbone_weights[key] = model_weights[key]
        else:
            other_weights[key] = model_weights[key]

    # Save the separated weights
    torch.save(backbone_weights, f'{output_dir}/backbone_weights.pth')
    torch.save(other_weights, f'{output_dir}/other_weights.pth')

    print("Weights have been separated and saved!")

if __name__ == "__main__":
    separate_and_save_weights('/mnt/zjy/model_merging/mergekit/optimization/ckpts/CCT/taskB_model_best_sd.pth', '/mnt/zjy/model_merging/mergekit/optimization/ckpts/CCT/taskB')