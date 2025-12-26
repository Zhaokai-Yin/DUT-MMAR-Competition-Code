import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from dataset import MultiModalDataset
from model import MultiModalTSM


def parse_args():
    parser = argparse.ArgumentParser(description='MMAR多模态行为识别测试')
    parser.add_argument('--data_root', type=str, default='MMAR/test_200',
                       help='测试数据根目录')
    parser.add_argument('--video_list', type=str, default='MMAR/test_200/test_videofolder_200.txt',
                       help='视频列表文件')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--num_segments', type=int, default=8,
                       help='视频分段数量')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--num_class', type=int, default=20,
                       help='类别数量')
    parser.add_argument('--output', type=str, default='predictions.npy',
                       help='预测结果保存路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    return parser.parse_args()


def test(model, dataloader, device):
    """测试模型"""
    model.eval()
    all_predictions = []
    all_video_ids = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Test]')
        for (rgb_input, depth_input, ir_input), video_ids in pbar:
            rgb_input = rgb_input.to(device)
            depth_input = depth_input.to(device)
            ir_input = ir_input.to(device)
            
            outputs = model(rgb_input, depth_input, ir_input)
            probabilities = torch.softmax(outputs, dim=1)
            
            all_predictions.append(probabilities.cpu().numpy())
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
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 获取训练参数
    if 'args' in checkpoint:
        train_args = checkpoint['args']
        base_model = train_args.base_model
        fusion_method = train_args.fusion_method
    else:
        # 默认参数
        base_model = 'resnet18'
        fusion_method = 'late'
    
    # 创建模型
    model = MultiModalTSM(
        num_class=args.num_class,
        num_segments=args.num_segments,
        base_model=base_model,
        fusion_method=fusion_method
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型: {args.checkpoint}")
    
    # 测试
    print("开始测试...")
    predictions, video_ids = test(model, test_loader, device)
    
    # 保存预测结果
    np.save(args.output, predictions)
    print(f"预测结果已保存到: {args.output}")
    print(f"预测结果形状: {predictions.shape}")
    print(f"视频ID数量: {len(video_ids)}")


if __name__ == '__main__':
    main()




