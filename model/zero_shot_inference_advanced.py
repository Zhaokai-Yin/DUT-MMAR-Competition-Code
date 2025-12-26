"""
高级零样本推理 - 使用更强的backbone
支持多种强大的预训练模型
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from dataset import MultiModalDataset
import torchvision.models.video as video_models
import torchvision.models as models


class StrongFeatureExtractor(nn.Module):
    """使用更强的预训练模型提取特征"""
    def __init__(self, model_name='efficientnet_b7'):
        super(StrongFeatureExtractor, self).__init__()
        self.model_name = model_name
        
        if model_name.startswith('efficientnet'):
            # EfficientNet系列（B7最强）
            if model_name == 'efficientnet_b7':
                self.model = models.efficientnet_b7(pretrained=True)
                self.feature_dim = 2560
            elif model_name == 'efficientnet_b4':
                self.model = models.efficientnet_b4(pretrained=True)
                self.feature_dim = 1792
            elif model_name == 'efficientnet_v2_l':
                self.model = models.efficientnet_v2_l(pretrained=True)
                self.feature_dim = 1280
            elif model_name == 'efficientnet_v2_m':
                self.model = models.efficientnet_v2_m(pretrained=True)
                self.feature_dim = 1280
            else:
                self.model = models.efficientnet_v2_s(pretrained=True)
                self.feature_dim = 1280
            
            # 移除分类层
            self.model.classifier = nn.Identity()
            
        elif model_name.startswith('convnext'):
            # ConvNeXt系列（最新的CNN架构）
            if model_name == 'convnext_large':
                self.model = models.convnext_large(pretrained=True)
                self.feature_dim = 1536
            elif model_name == 'convnext_base':
                self.model = models.convnext_base(pretrained=True)
                self.feature_dim = 1024
            elif model_name == 'convnext_small':
                self.model = models.convnext_small(pretrained=True)
                self.feature_dim = 768
            else:
                self.model = models.convnext_tiny(pretrained=True)
                self.feature_dim = 768
            
            self.model.classifier = nn.Identity()
            
        elif model_name.startswith('resnet'):
            # ResNet系列（ResNet152最强）
            if model_name == 'resnet152':
                self.model = models.resnet152(pretrained=True)
                self.feature_dim = 2048
            elif model_name == 'resnet101':
                self.model = models.resnet101(pretrained=True)
                self.feature_dim = 2048
            else:
                self.model = models.resnet50(pretrained=True)
                self.feature_dim = 2048
            
            self.model.fc = nn.Identity()
            
        elif model_name == 'r2plus1d_18':
            # R(2+1)D视频模型
            self.model = video_models.r2plus1d_18(pretrained=True)
            self.model.fc = nn.Identity()
            self.feature_dim = 512
            self.required_frames = 16
            
        elif model_name == 'r3d_18':
            # R3D视频模型
            self.model = video_models.r3d_18(pretrained=True)
            self.model.fc = nn.Identity()
            self.feature_dim = 512
            self.required_frames = 16
            
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model.eval()
    
    def forward(self, x):
        # x: [B, C, T, H, W] 或 [B, C, H, W]
        if hasattr(self, 'required_frames'):
            # 视频模型需要处理帧数
            B, C, T, H, W = x.size()
            if T < self.required_frames:
                repeat_times = (self.required_frames + T - 1) // T
                x = x.repeat(1, 1, repeat_times, 1, 1)
                x = x[:, :, :self.required_frames, :, :]
            elif T > self.required_frames:
                indices = torch.linspace(0, T - 1, self.required_frames).long()
                x = x[:, :, indices, :, :]
            
            with torch.no_grad():
                features = self.model(x)
        else:
            # 2D模型：需要处理时序
            B, C, T, H, W = x.size()
            # 重塑为 [B*T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B * T, C, H, W)
            
            with torch.no_grad():
                features = self.model(x)
                # 重塑回 [B, T, feature_dim]
                features = features.view(B, T, -1)
                # 时序平均池化
                features = features.mean(dim=1)
        
        return features


class MultiModalStrongExtractor(nn.Module):
    """多模态强特征提取器"""
    def __init__(self, model_name='efficientnet_b7'):
        super(MultiModalStrongExtractor, self).__init__()
        self.rgb_extractor = StrongFeatureExtractor(model_name)
        self.depth_extractor = StrongFeatureExtractor(model_name)
        self.ir_extractor = StrongFeatureExtractor(model_name)
        self.feature_dim = self.rgb_extractor.feature_dim * 3
    
    def forward(self, rgb_input, depth_input, ir_input):
        rgb_feat = self.rgb_extractor(rgb_input)
        depth_feat = self.depth_extractor(depth_input)
        ir_feat = self.ir_extractor(ir_input)
        # 拼接特征
        combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return combined_feat


def extract_features(model, dataloader, device, is_test_mode=False):
    """提取所有样本的特征"""
    model.eval()
    all_features = []
    all_labels = []
    all_video_ids = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='提取特征')
        for batch in pbar:
            # 处理batch格式（可能是tuple或list）
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                modalities, labels_or_ids = batch
                
                # 解包模态数据
                if isinstance(modalities, (tuple, list)) and len(modalities) == 3:
                    rgb_input, depth_input, ir_input = modalities
                else:
                    raise ValueError(f"意外的模态数据格式: {type(modalities)}")
                
                if is_test_mode:
                    # 测试模式：第二个元素是video_ids
                    video_ids = labels_or_ids
                    labels = None
                else:
                    # 训练模式：第二个元素是labels
                    labels = labels_or_ids
                    video_ids = None
            else:
                raise ValueError(f"意外的batch格式: {type(batch)}, 长度: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
            
            rgb_input = rgb_input.to(device, non_blocking=True)
            depth_input = depth_input.to(device, non_blocking=True)
            ir_input = ir_input.to(device, non_blocking=True)
            
            features = model(rgb_input, depth_input, ir_input)
            features = features.cpu().numpy()
            
            all_features.append(features)
            if labels is not None:
                all_labels.extend(labels.numpy())
            if video_ids is not None:
                all_video_ids.extend(video_ids.numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    
    if all_labels:
        all_labels = np.array(all_labels)
        return all_features, all_labels, None
    elif all_video_ids:
        return all_features, None, np.array(all_video_ids)
    else:
        return all_features, None, None


def main():
    parser = argparse.ArgumentParser(description='高级零样本推理 - 使用更强的backbone')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--model_name', type=str, default='efficientnet_b7',
                       choices=['efficientnet_b7', 'efficientnet_b4', 'efficientnet_v2_l', 
                               'efficientnet_v2_m', 'convnext_large', 'convnext_base',
                               'resnet152', 'resnet101', 'r2plus1d_18', 'r3d_18'],
                       help='预训练模型（efficientnet_b7最强）')
    parser.add_argument('--classifier', type=str, default='logistic',
                       choices=['logistic', 'svm', 'knn'],
                       help='分类器类型')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_segments', type=int, default=8,
                       help='视频分段数量')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据变换（根据模型调整）
    if args.model_name.startswith('efficientnet') or args.model_name.startswith('convnext'):
        # EfficientNet和ConvNeXt使用标准ImageNet归一化
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 视频模型使用Kinetics归一化
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                               std=[0.22803, 0.22145, 0.216989])
        ])
    
    # 创建特征提取模型
    print(f"加载预训练模型: {args.model_name}")
    feature_extractor = MultiModalStrongExtractor(args.model_name).to(device)
    feature_extractor.eval()
    
    # 提取训练集特征
    print("\n提取训练集特征...")
    train_dataset = MultiModalDataset(
        args.data_root_train, args.video_list_train,
        num_segments=args.num_segments,
        transform=transform,
        random_shift=False,
        test_mode=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    train_features, train_labels, _ = extract_features(
        feature_extractor, train_loader, device, is_test_mode=False
    )
    print(f"训练特征形状: {train_features.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    
    # 训练分类器
    print(f"\n训练{args.classifier}分类器...")
    if args.classifier == 'logistic':
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, multi_class='multinomial', 
                                     solver='lbfgs', C=1.0))
        ])
    elif args.classifier == 'svm':
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, C=1.0, gamma='scale'))
        ])
    else:  # knn
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance'))
        ])
    
    classifier.fit(train_features, train_labels)
    print("分类器训练完成！")
    
    # 提取测试集特征
    print("\n提取测试集特征...")
    test_dataset = MultiModalDataset(
        args.data_root_test, args.video_list_test,
        num_segments=args.num_segments,
        transform=transform,
        random_shift=False,
        test_mode=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    test_features, _, test_video_ids = extract_features(
        feature_extractor, test_loader, device, is_test_mode=True
    )
    print(f"测试特征形状: {test_features.shape}")
    
    # 预测
    print("\n进行预测...")
    predictions_proba = classifier.predict_proba(test_features)
    
    # 获取Top-5预测
    top5_indices = np.argsort(predictions_proba, axis=1)[:, -5:][:, ::-1]
    
    # 按video_id排序（确保test_video_ids不为None）
    if test_video_ids is None or len(test_video_ids) == 0:
        # 如果没有video_ids，从数据集中获取
        print("从数据集中获取video_ids...")
        test_video_ids = []
        for i in range(len(test_dataset)):
            _, video_id = test_dataset[i]
            test_video_ids.append(video_id)
        test_video_ids = np.array(test_video_ids)
    
    sorted_indices = np.argsort(test_video_ids)
    top5_indices = top5_indices[sorted_indices]
    test_video_ids = test_video_ids[sorted_indices]
    
    # 生成提交文件
    prediction_strings = []
    for pred in top5_indices:
        pred_str = ' '.join([str(int(cls)) for cls in pred])
        prediction_strings.append(pred_str)
    
    df = pd.DataFrame({
        'video_id': test_video_ids,
        'prediction': prediction_strings
    })
    
    df = df.sort_values('video_id').reset_index(drop=True)
    df.to_csv(args.output, index=False, header=True)
    
    print(f"\n提交文件已保存到: {args.output}")
    print(f"文件包含 {len(df)} 行数据")
    print("\n前10行预览:")
    print(df.head(10).to_string(index=False))
    
    # 保存分类器（可选）
    classifier_path = args.output.replace('.csv', '_classifier.pkl')
    joblib.dump(classifier, classifier_path)
    print(f"\n分类器已保存到: {classifier_path}")


if __name__ == '__main__':
    main()
