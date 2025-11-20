import torch
import timm
import torchmetrics
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader
from timm.data import create_dataset, create_loader
import os
import sys
from src import cct_14_7x2_224
import argparse
import json

def evaluation(model_path, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_map = {}
    class_map_path = '/mnt/zjy/model_merging/ADmodels/Compact-Transformers/dataset/train_maps/class_map_front_full1000_idx2cls.txt'
    with open(class_map_path, 'r') as f:
        for line in f:
            idx, dict_name = line.strip().split()
            class_map[dict_name] = int(idx)
            
    if dataset == 'front':
        dataset_path = '/mnt/zjy/model_merging/ADmodels/Compact-Transformers/dataset/split_imagenet/ImageNet1k_front500'
    elif dataset == 'back':
        dataset_path = '/mnt/zjy/model_merging/ADmodels/Compact-Transformers/dataset/split_imagenet/ImageNet1k_back500'
    test_dataset = create_dataset(
        'imagenet',
        root=dataset_path,  # 数据集目录
        split='validation', 
        is_training=False,  # 设置为False来加载测试集
        batch_size=128,
        class_map=class_map  # 使用自定义的class_map
    )
    test_loader = create_loader(
        test_dataset,
        input_size=(3, 224, 224),  # 输入图像大小
        batch_size=128,  # 设置批次大小
        is_training=False,  # 设置为False来加载测试集
        use_prefetcher=False,  # 不使用预取器
        interpolation='bicubic',  # 设置插值方法
        mean=(0.485, 0.456, 0.406),  # 设置均值
        std=(0.229, 0.224, 0.225),  # 设置标准差
        num_workers=8,  # 设置工作线程数
        distributed=False,  # 不使用分布式训练
        crop_pct=0.9,  # 设置裁剪比例
        pin_memory=False,  # 不使用pin_memory
    )
    model = cct_14_7x2_224(pretrained=False, num_classes=1000)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    top1_accuracy_metric = Accuracy(num_classes=1000, task='multiclass', top_k=1).to(device)
    top5_accuracy_metric = Accuracy(num_classes=1000, task='multiclass', top_k=5).to(device)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            top1_accuracy_metric.update(outputs, labels)
            top5_accuracy_metric.update(outputs, labels)
            
    top1_accuracy = top1_accuracy_metric.compute()
    top5_accuracy = top5_accuracy_metric.compute()
    
    result = {
        "top1_accuracy": top1_accuracy.item(),
        "top5_accuracy": top5_accuracy.item()
    }
    with open(f"/mnt/zjy/model_merging/mergekit/optimization/scores/CCT_scores/eval_{dataset}_result.json", "w") as f:
        json.dump(result, f, indent=4)
    
    return top5_accuracy.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate merged model on dataset")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint file')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset root directory')
    args = parser.parse_args()

    top5_acc = evaluation(args.model_path, args.dataset)