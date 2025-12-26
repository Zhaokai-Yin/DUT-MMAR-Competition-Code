import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

from dataset import MultiModalDataset
from model import MultiModalTSM


def parse_args():
    parser = argparse.ArgumentParser(description='MMAR多模态行为识别训练（8GB显存优化版）')
    parser.add_argument('--data_root', type=str, default='MMAR/train_500',
                       help='训练数据根目录')
    parser.add_argument('--video_list', type=str, default='MMAR/train_500/train_videofolder_500.txt',
                       help='视频列表文件')
    parser.add_argument('--num_segments', type=int, default=8,
                       help='视频分段数量')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小（8GB显存建议1-2）')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='梯度累积步数（实际batch_size = batch_size * accumulation_steps）')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--num_class', type=int, default=20,
                       help='类别数量')
    parser.add_argument('--base_model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50'],
                       help='基础模型（8GB显存建议resnet18）')
    parser.add_argument('--fusion_method', type=str, default='late',
                       choices=['early', 'late', 'attention'],
                       help='融合方法')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--resume', type=str, default='',
                       help='恢复训练的模型路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练（节省显存）')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='数据加载线程数（减少内存占用）')
    return parser.parse_args()


def split_train_val(data_root, video_list_file, val_ratio=0.2, random_seed=42):
    """划分训练集和验证集"""
    all_videos = []
    with open(video_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
        train_vids, val_vids = train_test_split(
            videos, test_size=val_ratio, random_state=random_seed
        )
        train_videos.extend(train_vids)
        val_videos.extend(val_vids)
    
    train_list_file = os.path.join(data_root, 'train_list.txt')
    val_list_file = os.path.join(data_root, 'val_list.txt')
    
    with open(train_list_file, 'w') as f:
        for line in train_videos:
            f.write(line + '\n')
    
    with open(val_list_file, 'w') as f:
        for line in val_videos:
            f.write(line + '\n')
    
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
        
        # 混合精度训练
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(rgb_input, depth_input, ir_input)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # 梯度累积
            
            scaler.scale(loss).backward()
        else:
            outputs = model(rgb_input, depth_input, ir_input)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
        
        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 梯度累积
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            # 清理缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    # 处理最后一个不完整的batch
    if len(dataloader) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


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
            
            # 定期清理缓存
            if total % 20 == 0:
                torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 划分训练集和验证集
    print("划分训练集和验证集...")
    train_list_file, val_list_file = split_train_val(
        args.data_root, args.video_list, args.val_ratio
    )
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
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
    
    # 创建数据加载器（减少num_workers以节省内存）
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
    model = MultiModalTSM(
        num_class=args.num_class,
        num_segments=args.num_segments,
        base_model=args.base_model,
        fusion_method=args.fusion_method
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr * args.accumulation_steps,  # 考虑梯度累积
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 混合精度训练
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("启用混合精度训练")
    
    # 恢复训练
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        print(f"从epoch {start_epoch}恢复训练")
    
    print(f"\n训练配置:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  梯度累积步数: {args.accumulation_steps}")
    print(f"  有效批次大小: {args.batch_size * args.accumulation_steps}")
    print(f"  数据加载线程数: {args.num_workers}")
    print(f"  混合精度训练: {'是' if args.use_amp else '否'}")
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            args.accumulation_steps, args.use_amp, scaler
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.use_amp)
        
        # 更新学习率
        scheduler.step()
        
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'  学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型
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
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()




