"""
ConvNeXt Large微调脚本（目标95+分）
基于92.54分的成功配置，进行端到端微调
这是最有可能达到95+分的方法
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
import gc

from dataset import MultiModalDataset
import torchvision.models as models


class ConvNeXtLargeTSM(nn.Module):
    """基于ConvNeXt Large的TSM模型（用于微调）"""
    def __init__(self, num_class, num_segments=8, dropout=0.5):
        super(ConvNeXtLargeTSM, self).__init__()
        self.num_segments = num_segments
        
        # 加载预训练ConvNeXt Large
        try:
            self.base_model = models.convnext_v2_large(pretrained=True)
            feature_dim = 1536
        except:
            self.base_model = models.convnext_large(pretrained=True)
            feature_dim = 1536
        
        # 移除分类层
        self.base_model.classifier = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_class)
    
    def forward(self, input):
        # input: [B, C, T, H, W]
        B, C, T, H, W = input.size()
        input = input.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        input = input.view(B * T, C, H, W)  # [B*T, C, H, W]
        
        base_out = self.base_model(input)  # [B*T, feature_dim]
        base_out = base_out.view(B, T, -1)  # [B, T, feature_dim]
        
        # 时序聚合（mean，与92.54分一致）
        output = base_out.mean(dim=1)  # [B, feature_dim]
        
        output = self.dropout(output)
        output = self.fc(output)  # [B, num_class]
        return output


class MultiModalConvNeXtLarge(nn.Module):
    """多模态ConvNeXt Large模型（用于微调）"""
    def __init__(self, num_class, num_segments=8, fusion_method='late', dropout=0.5):
        super(MultiModalConvNeXtLarge, self).__init__()
        self.fusion_method = fusion_method
        
        # 创建三个模态的模型
        self.rgb_model = ConvNeXtLargeTSM(num_class, num_segments, dropout)
        self.depth_model = ConvNeXtLargeTSM(num_class, num_segments, dropout)
        self.ir_model = ConvNeXtLargeTSM(num_class, num_segments, dropout)
        
        if fusion_method == 'late':
            # 晚期融合：三个模态分别预测，然后加权平均
            self.fusion_weight = nn.Parameter(torch.ones(3) / 3)
        elif fusion_method == 'concat':
            # 早期融合：拼接特征
            self.fc = nn.Linear(1536 * 3, num_class)
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
    
    def forward(self, rgb_input, depth_input, ir_input):
        if self.fusion_method == 'late':
            # 晚期融合（与92.54分一致）
            rgb_logits = self.rgb_model(rgb_input)
            depth_logits = self.depth_model(depth_input)
            ir_logits = self.ir_model(ir_input)
            
            # 加权平均
            weights = torch.softmax(self.fusion_weight, dim=0)
            output = weights[0] * rgb_logits + weights[1] * depth_logits + weights[2] * ir_logits
            return output
        else:
            # 早期融合
            rgb_feat = self.rgb_model.base_model(rgb_input)
            depth_feat = self.depth_model.base_model(depth_input)
            ir_feat = self.ir_model.base_model(ir_input)
            combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
            return self.fc(combined_feat)


def split_train_val(data_root, video_list_file, val_ratio=0.2):
    """划分训练集和验证集"""
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
    parser = argparse.ArgumentParser(description='ConvNeXt Large微调（目标95+分）')
    parser.add_argument('--data_root', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小（ConvNeXt Large建议1-2）')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='梯度累积步数（实际batch_size = batch_size × accumulation_steps）')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数（建议20-30轮）')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='学习率（预训练模型建议较小）')
    parser.add_argument('--num_class', type=int, default=20)
    parser.add_argument('--fusion_method', type=str, default='late',
                       choices=['late', 'concat'],
                       help='融合方法（late与92.54分一致）')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--save_dir', type=str, default='checkpoints_convnext_large')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练（推荐）')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("划分训练集和验证集...")
    train_list_file, val_list_file = split_train_val(args.data_root, args.video_list, args.val_ratio)
    
    # 数据增强（训练时使用更强的增强）
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证时使用与92.54分一致的变换
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
    print("\n创建ConvNeXt Large模型...")
    model = MultiModalConvNeXtLarge(
        num_class=args.num_class,
        num_segments=args.num_segments,
        fusion_method=args.fusion_method
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr * args.accumulation_steps,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("启用混合精度训练")
    
    best_val_acc = 0.0
    
    print(f"\n训练配置:")
    print(f"  Backbone: ConvNeXt Large")
    print(f"  批次大小: {args.batch_size} × {args.accumulation_steps} = {args.batch_size * args.accumulation_steps}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  融合方法: {args.fusion_method}")
    
    print("\n开始训练...")
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
    
    print("\n训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()



