"""
使用超强backbone生成提交文件
支持ConvNeXt、RegNet、ResNeXt、Wide ResNet、EfficientNet-V2、ViT等最强backbone
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import MultiModalDataset
from model_ultra import MultiModalUltra


def parse_args():
    parser = argparse.ArgumentParser(description='生成提交文件 - 超强backbone')
    parser.add_argument('--data_root', type=str, default='MMAR/test_200',
                       help='测试数据根目录')
    parser.add_argument('--video_list', type=str, default='MMAR/test_200/test_videofolder_200.txt',
                       help='视频列表文件')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_ultra.pth',
                       help='模型检查点路径')
    parser.add_argument('--num_segments', type=int, default=8,
                       help='视频分段数量')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--num_class', type=int, default=20,
                       help='类别数量')
    parser.add_argument('--backbone', type=str, default='convnext',
                       choices=['convnext', 'regnet', 'resnext', 'wide_resnet', 
                               'efficientnet_v2', 'vit'],
                       help='backbone类型')
    parser.add_argument('--model_size', type=str, default='base',
                       help='模型大小')
    parser.add_argument('--fusion_method', type=str, default='late',
                       choices=['late', 'attention'],
                       help='融合方法')
    parser.add_argument('--output', type=str, default='submission_ultra.csv',
                       help='输出CSV文件路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    return parser.parse_args()


def predict_top5(model, dataloader, device):
    """预测Top-5类别"""
    model.eval()
    all_predictions = []
    all_video_ids = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Predict]')
        for (rgb_input, depth_input, ir_input), video_ids in pbar:
            rgb_input = rgb_input.to(device, non_blocking=True)
            depth_input = depth_input.to(device, non_blocking=True)
            ir_input = ir_input.to(device, non_blocking=True)
            
            outputs = model(rgb_input, depth_input, ir_input)
            # 获取Top-5预测
            _, top5_indices = torch.topk(outputs, k=5, dim=1)
            top5_indices = top5_indices.cpu().numpy()
            
            all_predictions.append(top5_indices)
            all_video_ids.extend(video_ids.numpy())
    
    # 合并所有预测结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # 按video_id排序
    sorted_indices = np.argsort(all_video_ids)
    all_predictions = all_predictions[sorted_indices]
    all_video_ids = np.array(all_video_ids)[sorted_indices]
    
    return all_predictions, all_video_ids


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据变换
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建测试数据集
    test_dataset = MultiModalDataset(
        args.data_root, args.video_list,
        num_segments=args.num_segments,
        transform=test_transform,
        random_shift=False,
        test_mode=True
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 加载模型
    if not os.path.exists(args.checkpoint):
        print(f"警告: 模型文件不存在: {args.checkpoint}")
        print("使用预训练权重创建新模型（零样本模式）")
        
        # 创建模型（使用预训练权重）
        model = MultiModalUltra(
            num_class=args.num_class,
            num_segments=args.num_segments,
            backbone=args.backbone,
            model_size=args.model_size,
            fusion_method=args.fusion_method
        ).to(device)
        
        print(f"使用backbone: {args.backbone} ({args.model_size})")
        print("注意: 这是零样本模式，模型使用预训练权重，未经过微调")
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # 获取训练参数
        if 'args' in checkpoint:
            train_args = checkpoint['args']
            backbone = train_args.backbone if hasattr(train_args, 'backbone') else args.backbone
            model_size = train_args.model_size if hasattr(train_args, 'model_size') else args.model_size
            fusion_method = train_args.fusion_method if hasattr(train_args, 'fusion_method') else args.fusion_method
        else:
            backbone = args.backbone
            model_size = args.model_size
            fusion_method = args.fusion_method
        
        # 创建模型
        model = MultiModalUltra(
            num_class=args.num_class,
            num_segments=args.num_segments,
            backbone=backbone,
            model_size=model_size,
            fusion_method=fusion_method
        ).to(device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型: {args.checkpoint}")
        print(f"Backbone: {backbone} ({model_size}), 融合方法: {fusion_method}")
    
    # 预测
    print("开始预测...")
    predictions, video_ids = predict_top5(model, test_loader, device)
    
    # 验证预测结果数量
    if len(predictions) != 200:
        print(f"警告: 预测结果数量 ({len(predictions)}) 与测试集数量 (200) 不一致！")
    
    # 格式化预测结果为字符串
    prediction_strings = []
    for pred in predictions:
        # 将类别索引转换为字符串，用空格分隔
        pred_str = ' '.join([str(int(cls)) for cls in pred])
        prediction_strings.append(pred_str)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'video_id': video_ids,
        'prediction': prediction_strings
    })
    
    # 确保按video_id排序
    df = df.sort_values('video_id').reset_index(drop=True)
    
    # 保存为CSV文件
    df.to_csv(args.output, index=False, header=True)
    print(f"提交文件已保存到: {args.output}")
    print(f"文件包含 {len(df)} 行数据")
    
    # 显示前几行
    print("\n前10行预览:")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()






