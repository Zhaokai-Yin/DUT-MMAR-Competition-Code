"""
超级快速微调版本 - 使用最强backbone + 快速微调
5-10个epoch即可获得最佳效果！
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc

from dataset import MultiModalDataset
from model_advanced import MultiModalAdvanced


def parse_args():
    parser = argparse.ArgumentParser(description='超级快速微调 - 最强backbone + 快速微调')
    parser.add_argument('--data_root', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--backbone', type=str, default='efficientnet',
                       choices=['efficientnet', 'convnext', 'r2plus1d'])
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'medium', 'large'])
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10,
                       help='快速微调，10个epoch即可')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='较小的学习率用于微调')
    parser.add_argument('--num_class', type=int, default=20)
    parser.add_argument('--fusion_method', type=str, default='attention',
                       choices=['late', 'attention'],
                       help='使用attention融合效果更好')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--save_dir', type=str, default='checkpoints_super')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def split_train_val(data_root, video_list_file, val_ratio=0.2):
    all_videos = []
    with open(video_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                all_videos.append(line)
    
    videos_by_class = {}
    for line in all_videos:
        parts = line.split()
        if len(parts) >= 3:
            label = int(parts[2])
            if label not in videos_by_class:
                videos_by_class[label] = []
            videos_by_class[label].append(line)
    
    train_videos = []
    val_videos = []
    for label, videos in videos_by_class.items():
        train_vids, val_vids = train_test_split(videos, test_size=val_ratio, random_state=42)
        train_videos.extend(train_vids)
        val_videos.extend(val_vids)
    
    train_list_file = os.path.join(data_root, 'train_list.txt')
    val_list_file = os.path.join(data_root, 'val_list.txt')
    
    with open(train_list_file, 'w') as f:
        f.write('\n'.join(train_videos) + '\n')
    with open(val_list_file, 'w') as f:
        f.write('\n'.join(val_videos) + '\n')
    
    print(f"训练集: {len(train_videos)} 个视频")
    print(f"验证集: {len(val_videos)} 个视频")
    return train_list_file, val_list_file


def train_epoch(model, dataloader, criterion, optimizer, device, epoch,
                accumulation_steps=1, use_amp=False, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, ((rgb_input, depth_input, ir_input), labels) in enumerate(pbar):
        rgb_input = rgb_input.to(device, non_blocking=True)
        depth_input = depth_input.to(device, non_blocking=True)
        ir_input = ir_input.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(rgb_input, depth_input, ir_input)
                loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(rgb_input, depth_input, ir_input)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
        
        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    if len(dataloader) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    return running_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Val]')
        for (rgb_input, depth_input, ir_input), labels in pbar:
            rgb_input = rgb_input.to(device, non_blocking=True)
            depth_input = depth_input.to(device, non_blocking=True)
            ir_input = ir_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(rgb_input, depth_input, ir_input)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(rgb_input, depth_input, ir_input)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    return running_loss / len(dataloader), 100 * correct / total


def main():
    args = parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("划分训练集和验证集...")
    train_list_file, val_list_file = split_train_val(args.data_root, args.video_list, args.val_ratio)
    
    # 强数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MultiModalDataset(
        args.data_root, train_list_file,
        num_segments=args.num_segments,
        transform=train_transform,
        random_shift=True,
        test_mode=False
    )
    
    val_dataset = MultiModalDataset(
        args.data_root, val_list_file,
        num_segments=args.num_segments,
        transform=val_transform,
        random_shift=False,
        test_mode=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=True if args.num_workers > 0 else False
    )
    
    # 创建模型
    model = MultiModalAdvanced(
        num_class=args.num_class,
        num_segments=args.num_segments,
        backbone=args.backbone,
        model_size=args.model_size,
        fusion_method=args.fusion_method
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 使用AdamW优化器（微调效果更好）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing提升泛化
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr * args.accumulation_steps,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # 使用Cosine退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("启用混合精度训练")
    
    best_val_acc = 0.0
    
    print(f"\n训练配置:")
    print(f"  Backbone: {args.backbone} ({args.model_size})")
    print(f"  融合方法: {args.fusion_method}")
    print(f"  批次大小: {args.batch_size} × {args.accumulation_steps} = {args.batch_size * args.accumulation_steps}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率: {args.lr}")
    
    print("\n开始快速微调...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            args.accumulation_steps, args.use_amp, scaler
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.use_amp)
        
        scheduler.step()
        
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'  学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  ✓ 保存最佳模型 (验证准确率: {best_val_acc:.2f}%)')
        
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n快速微调完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    # 生成测试集预测
    print("\n生成测试集预测...")
    test_transform = val_transform
    test_dataset = MultiModalDataset(
        args.data_root_test, args.video_list_test,
        num_segments=args.num_segments,
        transform=test_transform,
        random_shift=False,
        test_mode=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_predictions = []
    all_video_ids = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='[Test]')
        for (rgb_input, depth_input, ir_input), video_ids in pbar:
            rgb_input = rgb_input.to(device, non_blocking=True)
            depth_input = depth_input.to(device, non_blocking=True)
            ir_input = ir_input.to(device, non_blocking=True)
            
            outputs = model(rgb_input, depth_input, ir_input)
            probabilities = torch.softmax(outputs, dim=1)
            
            all_predictions.append(probabilities.cpu().numpy())
            all_video_ids.extend(video_ids.numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # 获取Top-5预测
    top5_indices = np.argsort(all_predictions, axis=1)[:, -5:][:, ::-1]
    
    # 按video_id排序
    sorted_indices = np.argsort(all_video_ids)
    top5_indices = top5_indices[sorted_indices]
    all_video_ids = np.array(all_video_ids)[sorted_indices]
    
    # 生成提交文件
    prediction_strings = []
    for pred in top5_indices:
        pred_str = ' '.join([str(int(cls)) for cls in pred])
        prediction_strings.append(pred_str)
    
    df = pd.DataFrame({
        'video_id': all_video_ids,
        'prediction': prediction_strings
    })
    
    df = df.sort_values('video_id').reset_index(drop=True)
    df.to_csv('submission.csv', index=False, header=True)
    
    print(f"\n提交文件已保存到: submission.csv")
    print(f"文件包含 {len(df)} 行数据")
    print("\n前10行预览:")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()

