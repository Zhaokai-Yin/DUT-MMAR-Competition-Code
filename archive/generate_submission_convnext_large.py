"""
生成ConvNeXt Large微调模型的提交文件
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
from train_convnext_large_95plus import MultiModalConvNeXtLarge


def predict_top5(model, dataloader, device, tta_times=10):
    """预测Top-5类别（支持TTA）"""
    model.eval()
    all_predictions = []
    all_video_ids = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Predict]')
        for (rgb_input, depth_input, ir_input), video_ids in pbar:
            rgb_input = rgb_input.to(device)
            depth_input = depth_input.to(device)
            ir_input = ir_input.to(device)
            
            # TTA: 多次预测取平均
            all_outputs = []
            for _ in range(tta_times):
                outputs = model(rgb_input, depth_input, ir_input)
                all_outputs.append(torch.softmax(outputs, dim=1))
            
            # 平均概率
            avg_probs = torch.stack(all_outputs).mean(dim=0)
            
            # 获取Top-5预测
            _, top5_indices = torch.topk(avg_probs, k=5, dim=1)
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
    parser = argparse.ArgumentParser(description='生成ConvNeXt Large微调模型的提交文件')
    parser.add_argument('--data_root', type=str, default='MMAR/test_200',
                       help='测试数据根目录')
    parser.add_argument('--video_list', type=str, default='MMAR/test_200/test_videofolder_200.txt',
                       help='视频列表文件')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_convnext_large/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--num_segments', type=int, default=8,
                       help='视频分段数量')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小')
    parser.add_argument('--num_class', type=int, default=20,
                       help='类别数量')
    parser.add_argument('--output', type=str, default='submission_convnext_large_finetuned.csv',
                       help='输出CSV文件路径')
    parser.add_argument('--tta_times', type=int, default=10,
                       help='测试时增强次数')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据变换（与训练时验证集一致）
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
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # 加载模型
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"模型文件不存在: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 获取训练参数
    if 'args' in checkpoint:
        train_args = checkpoint['args']
        fusion_method = train_args.fusion_method
    else:
        fusion_method = 'late'
    
    # 创建模型
    model = MultiModalConvNeXtLarge(
        num_class=args.num_class,
        num_segments=args.num_segments,
        fusion_method=fusion_method
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型: {args.checkpoint}")
    print(f"最佳验证准确率: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    # 预测
    print(f"开始预测（TTA={args.tta_times}）...")
    predictions, video_ids = predict_top5(model, test_loader, device, args.tta_times)
    
    # 验证预测结果数量
    if len(predictions) != 200:
        print(f"警告: 预测结果数量 ({len(predictions)}) 与测试集数量 (200) 不一致！")
    
    # 格式化预测结果为字符串
    prediction_strings = []
    for pred in predictions:
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
    print(f"\n提交文件已保存到: {args.output}")
    print(f"文件包含 {len(df)} 行数据")
    
    # 显示前几行
    print("\n前10行预览:")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()



