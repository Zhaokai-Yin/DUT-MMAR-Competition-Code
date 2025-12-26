"""
使用高级模型生成提交文件
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
from model_advanced import MultiModalAdvanced


def parse_args():
    parser = argparse.ArgumentParser(description='生成提交文件 - 高级模型')
    parser.add_argument('--data_root', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_class', type=int, default=20)
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def predict_top5(model, dataloader, device):
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
            _, top5_indices = torch.topk(outputs, k=5, dim=1)
            top5_indices = top5_indices.cpu().numpy()
            
            all_predictions.append(top5_indices)
            all_video_ids.extend(video_ids.numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    sorted_indices = np.argsort(all_video_ids)
    all_predictions = all_predictions[sorted_indices]
    all_video_ids = np.array(all_video_ids)[sorted_indices]
    
    return all_predictions, all_video_ids


def main():
    args = parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = MultiModalDataset(
        args.data_root, args.video_list,
        num_segments=args.num_segments,
        transform=test_transform,
        random_shift=False,
        test_mode=True
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"模型文件不存在: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 获取训练参数
    if 'args' in checkpoint:
        train_args = checkpoint['args']
        backbone = train_args.backbone
        model_size = train_args.model_size
        fusion_method = train_args.fusion_method
        num_segments = train_args.num_segments
    else:
        # 默认参数
        backbone = 'efficientnet'
        model_size = 'small'
        fusion_method = 'late'
        num_segments = args.num_segments
    
    # 创建模型
    model = MultiModalAdvanced(
        num_class=args.num_class,
        num_segments=num_segments,
        backbone=backbone,
        model_size=model_size,
        fusion_method=fusion_method
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型: {args.checkpoint}")
    print(f"Backbone: {backbone} ({model_size}), 融合方法: {fusion_method}")
    
    print("开始预测...")
    predictions, video_ids = predict_top5(model, test_loader, device)
    
    if len(predictions) != 200:
        print(f"警告: 预测结果数量 ({len(predictions)}) 与测试集数量 (200) 不一致！")
    
    prediction_strings = []
    for pred in predictions:
        pred_str = ' '.join([str(int(cls)) for cls in pred])
        prediction_strings.append(pred_str)
    
    df = pd.DataFrame({
        'video_id': video_ids,
        'prediction': prediction_strings
    })
    
    df = df.sort_values('video_id').reset_index(drop=True)
    df.to_csv(args.output, index=False, header=True)
    
    print(f"提交文件已保存到: {args.output}")
    print(f"文件包含 {len(df)} 行数据")
    print("\n前10行预览:")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()



