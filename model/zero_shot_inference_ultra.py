"""
超强零样本推理 - 使用Vision Transformer和最新最强模型
支持ViT、ConvNeXt V2、Swin Transformer等最强backbone
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
import torchvision.models as models


class UltraStrongFeatureExtractor(nn.Module):
    """使用超强的预训练模型提取特征（ViT、ConvNeXt V2等）"""
    def __init__(self, model_name='vit_large_patch16_224'):
        super(UltraStrongFeatureExtractor, self).__init__()
        self.model_name = model_name
        
        if model_name.startswith('vit'):
            # Vision Transformer系列（最强）
            try:
                if model_name == 'vit_huge_patch14_224':
                    # ViT-Huge（最大最强）
                    try:
                        self.model = models.vit_h_14(pretrained=True)
                    except:
                        self.model = models.vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1')
                    self.feature_dim = 1280
                elif model_name == 'vit_large_patch16_224':
                    # ViT-Large（推荐）
                    try:
                        self.model = models.vit_l_16(pretrained=True)
                    except:
                        self.model = models.vit_l_16(weights='IMAGENET1K_SWAG_E2E_V1')
                    self.feature_dim = 1024
                elif model_name == 'vit_base_patch16_224':
                    try:
                        self.model = models.vit_b_16(pretrained=True)
                    except:
                        self.model = models.vit_b_16(weights='IMAGENET1K_V1')
                    self.feature_dim = 768
                else:
                    # 默认ViT-Large
                    try:
                        self.model = models.vit_l_16(pretrained=True)
                    except:
                        self.model = models.vit_l_16(weights='IMAGENET1K_SWAG_E2E_V1')
                    self.feature_dim = 1024
                
                # ViT的head是分类层，需要移除
                if hasattr(self.model, 'heads'):
                    self.model.heads = nn.Identity()
                elif hasattr(self.model, 'head'):
                    self.model.head = nn.Identity()
            except Exception as e:
                print(f"ViT模型加载失败: {e}")
                print("尝试使用ConvNeXt-Large作为备选...")
                self.model = models.convnext_large(pretrained=True)
                self.feature_dim = 1536
                self.model.classifier = nn.Identity()
            
        elif model_name.startswith('convnext'):
            # ConvNeXt V2系列（最新）
            if model_name == 'convnextv2_large':
                try:
                    self.model = models.convnext_v2_large(pretrained=True)
                    self.feature_dim = 1536
                except:
                    # 如果没有V2，使用V1
                    self.model = models.convnext_large(pretrained=True)
                    self.feature_dim = 1536
            elif model_name == 'convnextv2_base':
                try:
                    self.model = models.convnext_v2_base(pretrained=True)
                    self.feature_dim = 1024
                except:
                    self.model = models.convnext_base(pretrained=True)
                    self.feature_dim = 1024
            else:
                self.model = models.convnext_large(pretrained=True)
                self.feature_dim = 1536
            
            self.model.classifier = nn.Identity()
            
        elif model_name.startswith('swin'):
            # Swin Transformer系列
            try:
                if model_name == 'swin_large':
                    try:
                        self.model = models.swin_v2_l(pretrained=True)
                    except:
                        try:
                            self.model = models.swin_l(pretrained=True)
                        except:
                            self.model = models.swin_v2_l(weights='IMAGENET1K_V1')
                    self.feature_dim = 1536
                elif model_name == 'swin_base':
                    try:
                        self.model = models.swin_v2_b(pretrained=True)
                    except:
                        try:
                            self.model = models.swin_b(pretrained=True)
                        except:
                            self.model = models.swin_v2_b(weights='IMAGENET1K_V1')
                    self.feature_dim = 1024
                else:
                    try:
                        self.model = models.swin_v2_t(pretrained=True)
                    except:
                        try:
                            self.model = models.swin_t(pretrained=True)
                        except:
                            self.model = models.swin_v2_t(weights='IMAGENET1K_V1')
                    self.feature_dim = 768
                
                if hasattr(self.model, 'head'):
                    self.model.head = nn.Identity()
            except Exception as e:
                print(f"Swin模型加载失败: {e}")
                self.model = models.convnext_large(pretrained=True)
                self.feature_dim = 1536
                self.model.classifier = nn.Identity()
            
        elif model_name.startswith('regnet'):
            # RegNet系列（Facebook的强模型）
            if model_name == 'regnet_y_128gf':
                try:
                    self.model = models.regnet_y_128gf(pretrained=True)
                    self.feature_dim = 3024
                except:
                    self.model = models.regnet_y_32gf(pretrained=True)
                    self.feature_dim = 3712
            elif model_name == 'regnet_y_32gf':
                self.model = models.regnet_y_32gf(pretrained=True)
                self.feature_dim = 3712
            else:
                self.model = models.regnet_y_16gf(pretrained=True)
                self.feature_dim = 3024
            
            self.model.fc = nn.Identity()
            
        elif model_name == 'maxvit_t':
            # MaxViT（最新的强模型）
            try:
                self.model = models.maxvit_t(pretrained=True)
                self.feature_dim = 512
                self.model.classifier = nn.Identity()
            except:
                # 如果没有MaxViT，使用ConvNeXt
                print("MaxViT不可用，使用ConvNeXt-Large代替")
                self.model = models.convnext_large(pretrained=True)
                self.feature_dim = 1536
                self.model.classifier = nn.Identity()
                
        elif model_name.startswith('efficientnet'):
            # EfficientNet系列（作为备选）
            if model_name == 'efficientnet_b7':
                self.model = models.efficientnet_b7(pretrained=True)
                self.feature_dim = 2560
            else:
                self.model = models.efficientnet_v2_l(pretrained=True)
                self.feature_dim = 1280
            
            self.model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model.eval()
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.size()
        
        # 2D模型：需要处理时序
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


class MultiModalUltraExtractor(nn.Module):
    """多模态超强特征提取器"""
    def __init__(self, model_name='vit_large_patch16_224'):
        super(MultiModalUltraExtractor, self).__init__()
        self.rgb_extractor = UltraStrongFeatureExtractor(model_name)
        self.depth_extractor = UltraStrongFeatureExtractor(model_name)
        self.ir_extractor = UltraStrongFeatureExtractor(model_name)
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
    parser = argparse.ArgumentParser(description='超强零样本推理 - 使用ViT和最新最强模型')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--model_name', type=str, default='vit_large_patch16_224',
                       choices=['vit_huge_patch14_224', 'vit_large_patch16_224', 'vit_base_patch16_224',
                               'convnextv2_large', 'convnextv2_base', 'convnext_large',
                               'swin_large', 'swin_base', 'swin_tiny',
                               'regnet_y_128gf', 'regnet_y_32gf', 'regnet_y_16gf',
                               'maxvit_t', 'efficientnet_b7'],
                       help='预训练模型（vit_large_patch16_224推荐，vit_huge_patch14_224最强）')
    parser.add_argument('--classifier', type=str, default='svm',
                       choices=['logistic', 'svm', 'knn'],
                       help='分类器类型（SVM通常更准确）')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    # 数据变换（ViT使用224x224）
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建特征提取模型
    print(f"加载预训练模型: {args.model_name}")
    try:
        feature_extractor = MultiModalUltraExtractor(args.model_name).to(device)
        feature_extractor.eval()
        print(f"模型特征维度: {feature_extractor.feature_dim}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用ViT-Large作为备选...")
        feature_extractor = MultiModalUltraExtractor('vit_large_patch16_224').to(device)
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
    
    # 训练分类器（使用SVM可能更准确）
    print(f"\n训练{args.classifier}分类器...")
    if args.classifier == 'logistic':
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=3000, multi_class='multinomial', 
                                     solver='lbfgs', C=1.0, n_jobs=-1))
        ])
    elif args.classifier == 'svm':
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, C=2.0, gamma='scale', 
                       decision_function_shape='ovr'))
        ])
    else:  # knn
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))
        ])
    
    print("开始训练分类器（这可能需要几分钟）...")
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
    
    # 如果video_ids为None，从数据集获取
    if test_video_ids is None or len(test_video_ids) == 0:
        print("从数据集中获取video_ids...")
        test_video_ids = []
        for i in range(len(test_dataset)):
            _, video_id = test_dataset[i]
            test_video_ids.append(video_id)
        test_video_ids = np.array(test_video_ids)
    
    # 预测
    print("\n进行预测...")
    predictions_proba = classifier.predict_proba(test_features)
    
    # 获取Top-5预测
    top5_indices = np.argsort(predictions_proba, axis=1)[:, -5:][:, ::-1]
    
    # 按video_id排序
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
