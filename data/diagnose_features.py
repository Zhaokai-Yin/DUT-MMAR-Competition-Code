"""
诊断脚本：检查特征提取是否正确
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MultiModalDataset
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# 快速测试特征提取
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvNeXtLargeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.model = models.convnext_v2_large(pretrained=True)
            self.feature_dim = 1536
        except:
            self.model = models.convnext_large(pretrained=True)
            self.feature_dim = 1536
        self.model.classifier = nn.Identity()
        self.model.eval()
    
    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            features = self.model(x)
            features = features.view(B, T, -1)
            features = features.mean(dim=1)
        return features

class MultiModalExtractor(nn.Module):
    def __init__(self, use_l2_norm=True):
        super().__init__()
        self.rgb_extractor = ConvNeXtLargeExtractor()
        self.depth_extractor = ConvNeXtLargeExtractor()
        self.ir_extractor = ConvNeXtLargeExtractor()
        self.use_l2_norm = use_l2_norm
        self.feature_dim = 1536 * 3
    
    def forward(self, rgb_input, depth_input, ir_input):
        rgb_feat = self.rgb_extractor(rgb_input)
        depth_feat = self.depth_extractor(depth_input)
        ir_feat = self.ir_extractor(ir_input)
        
        if self.use_l2_norm:
            rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
            depth_feat = F.normalize(depth_feat, p=2, dim=1)
            ir_feat = F.normalize(ir_feat, p=2, dim=1)
        
        combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return combined_feat

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("加载数据集...")
train_dataset = MultiModalDataset(
    'MMAR/train_500', 'MMAR/train_500/train_videofolder_500.txt',
    num_segments=8, transform=transform, random_shift=False, test_mode=False
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

print("\n测试特征提取（无L2归一化）...")
model_no_norm = MultiModalExtractor(use_l2_norm=False).to(device)
model_no_norm.eval()

features_no_norm = []
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        if i >= 5:  # 只测试前5个样本
            break
        modalities, labels = batch
        rgb_input, depth_input, ir_input = modalities
        rgb_input = rgb_input.to(device)
        depth_input = depth_input.to(device)
        ir_input = ir_input.to(device)
        
        feat = model_no_norm(rgb_input, depth_input, ir_input)
        features_no_norm.append(feat.cpu().numpy())
        
        # 提取单个特征用于打印
        rgb_feat = model_no_norm.rgb_extractor(rgb_input)
        depth_feat = model_no_norm.depth_extractor(depth_input)
        ir_feat = model_no_norm.ir_extractor(ir_input)
        
        print(f"  样本 {i+1}:")
        print(f"    RGB特征范围: [{rgb_feat.min():.4f}, {rgb_feat.max():.4f}], 均值: {rgb_feat.mean():.4f}, 标准差: {rgb_feat.std():.4f}")
        print(f"    Depth特征范围: [{depth_feat.min():.4f}, {depth_feat.max():.4f}], 均值: {depth_feat.mean():.4f}, 标准差: {depth_feat.std():.4f}")
        print(f"    IR特征范围: [{ir_feat.min():.4f}, {ir_feat.max():.4f}], 均值: {ir_feat.mean():.4f}, 标准差: {ir_feat.std():.4f}")
        print(f"    组合特征范围: [{feat.min():.4f}, {feat.max():.4f}], 均值: {feat.mean():.4f}, 标准差: {feat.std():.4f}")

print("\n测试特征提取（有L2归一化）...")
model_with_norm = MultiModalExtractor(use_l2_norm=True).to(device)
model_with_norm.eval()

features_with_norm = []
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        modalities, labels = batch
        rgb_input, depth_input, ir_input = modalities
        rgb_input = rgb_input.to(device)
        depth_input = depth_input.to(device)
        ir_input = ir_input.to(device)
        
        feat = model_with_norm(rgb_input, depth_input, ir_input)
        features_with_norm.append(feat.cpu().numpy())
        
        # 提取单个特征用于打印
        rgb_feat = model_with_norm.rgb_extractor(rgb_input)
        depth_feat = model_with_norm.depth_extractor(depth_input)
        ir_feat = model_with_norm.ir_extractor(ir_input)
        rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
        depth_feat = F.normalize(depth_feat, p=2, dim=1)
        ir_feat = F.normalize(ir_feat, p=2, dim=1)
        
        print(f"  样本 {i+1}:")
        print(f"    RGB特征L2范数: {torch.norm(rgb_feat, p=2, dim=1).mean():.4f}")
        print(f"    Depth特征L2范数: {torch.norm(depth_feat, p=2, dim=1).mean():.4f}")
        print(f"    IR特征L2范数: {torch.norm(ir_feat, p=2, dim=1).mean():.4f}")
        print(f"    组合特征范围: [{feat.min():.4f}, {feat.max():.4f}], 均值: {feat.mean():.4f}, 标准差: {feat.std():.4f}")

print("\n诊断完成！")

