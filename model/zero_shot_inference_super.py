"""
超级无敌爆炸厉害版本 - 集成多个最强预训练模型
使用模型ensemble + 最强特征融合 + 优化分类器 + TTA + 多尺度 + 集成学习
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

from dataset import MultiModalDataset
import torchvision.models as models

# 尝试导入timm（如果有的话，包含更多最强模型）
try:
    import timm
    HAS_TIMM = True
except:
    HAS_TIMM = False
    print("timm库未安装，将使用torchvision模型")


class SuperFeatureExtractor(nn.Module):
    """超级特征提取器 - 集成多个最强模型"""
    def __init__(self, model_names=['vit_l_16', 'convnext_large', 'efficientnet_b7']):
        super(SuperFeatureExtractor, self).__init__()
        self.model_names = model_names
        self.extractors = nn.ModuleList()
        self.feature_dims = []
        
        for model_name in model_names:
            extractor = self._create_extractor(model_name)
            self.extractors.append(extractor)
            self.feature_dims.append(extractor.feature_dim)
        
        self.total_feature_dim = sum(self.feature_dims)
    
    def _create_extractor(self, model_name):
        """创建单个特征提取器"""
        if model_name.startswith('vit'):
            # Vision Transformer
            if 'h' in model_name or 'huge' in model_name:
                try:
                    model = models.vit_h_14(pretrained=True)
                    feature_dim = 1280
                except:
                    model = models.vit_l_16(pretrained=True)
                    feature_dim = 1024
            elif 'l' in model_name or 'large' in model_name:
                try:
                    model = models.vit_l_16(pretrained=True)
                except:
                    model = models.vit_l_16(weights='IMAGENET1K_SWAG_E2E_V1')
                feature_dim = 1024
            else:
                model = models.vit_b_16(pretrained=True)
                feature_dim = 768
            
            if hasattr(model, 'heads'):
                model.heads = nn.Identity()
            elif hasattr(model, 'head'):
                model.head = nn.Identity()
            
            model_type = 'vit'  # 标记为ViT类型
            
        elif model_name.startswith('convnext'):
            # ConvNeXt
            if 'large' in model_name:
                try:
                    model = models.convnext_v2_large(pretrained=True)
                except:
                    model = models.convnext_large(pretrained=True)
                feature_dim = 1536
            else:
                model = models.convnext_base(pretrained=True)
                feature_dim = 1024
            
            model.classifier = nn.Identity()
            model_type = 'cnn'  # 标记为CNN类型
            
        elif model_name.startswith('efficientnet'):
            # EfficientNet
            if 'b7' in model_name:
                model = models.efficientnet_b7(pretrained=True)
                feature_dim = 2560
            else:
                model = models.efficientnet_v2_l(pretrained=True)
                feature_dim = 1280
            
            model.classifier = nn.Identity()
            model_type = 'cnn'  # 标记为CNN类型
            
        elif HAS_TIMM and model_name.startswith('timm_'):
            # 使用timm库的模型（如果有）
            timm_name = model_name.replace('timm_', '')
            try:
                model = timm.create_model(timm_name, pretrained=True, num_classes=0)
                feature_dim = model.num_features
                model_type = 'cnn' if 'vit' not in timm_name.lower() else 'vit'
            except:
                # 降级到ViT-Large
                model = models.vit_l_16(pretrained=True)
                feature_dim = 1024
                if hasattr(model, 'heads'):
                    model.heads = nn.Identity()
                model_type = 'vit'
        else:
            # 默认使用ViT-Large
            model = models.vit_l_16(pretrained=True)
            feature_dim = 1024
            if hasattr(model, 'heads'):
                model.heads = nn.Identity()
            model_type = 'vit'
        
        model.eval()
        extractor = nn.Module()
        extractor.model = model
        extractor.feature_dim = feature_dim
        extractor.model_type = model_type  # 保存模型类型
        return extractor
    
    def forward(self, x, use_multiscale=False):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        
        all_features = []
        with torch.no_grad():
            for extractor in self.extractors:
                if use_multiscale:
                    # 多尺度特征提取
                    feat_list = []
                    # 原始尺度
                    feat = extractor.model(x)
                    feat_list.append(feat)
                    
                    # 根据模型类型选择多尺度策略
                    model_type = getattr(extractor, 'model_type', 'cnn')
                    
                    if model_type == 'vit':
                        # ViT模型：只使用224x224，但可以使用不同的裁剪方式
                        # 中心裁剪
                        if H > 224:
                            x_center = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                            feat_center = extractor.model(x_center)
                            feat_list.append(feat_center)
                        # 如果已经是224，可以尝试轻微的数据增强（翻转等）
                    else:
                        # CNN模型（ConvNeXt、EfficientNet等）：可以使用多尺度
                        # 更大尺度 (256x256)
                        if H < 256:
                            x_large = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
                            feat_large = extractor.model(x_large)
                            feat_list.append(feat_large)
                        # 更小尺度 (192x192)
                        if H > 192:
                            x_small = F.interpolate(x, size=(192, 192), mode='bilinear', align_corners=False)
                            feat_small = extractor.model(x_small)
                            feat_list.append(feat_small)
                    
                    # 平均多尺度特征
                    feat = torch.stack(feat_list, dim=0).mean(dim=0)
                else:
                    feat = extractor.model(x)
                
                feat = feat.view(B, T, -1)
                # 使用多种时序聚合方式
                feat_mean = feat.mean(dim=1)  # 平均
                feat_max = feat.max(dim=1)[0]  # 最大
                feat_std = feat.std(dim=1)  # 标准差
                # 拼接多种时序特征
                feat = torch.cat([feat_mean, feat_max, feat_std], dim=1)
                all_features.append(feat)
        
        # 拼接所有特征
        combined = torch.cat(all_features, dim=1)
        return combined


class MultiModalSuperExtractor(nn.Module):
    """多模态超级特征提取器 - 集成多个最强模型"""
    def __init__(self, model_names=['vit_l_16', 'convnext_large', 'efficientnet_b7']):
        super(MultiModalSuperExtractor, self).__init__()
        self.rgb_extractor = SuperFeatureExtractor(model_names)
        self.depth_extractor = SuperFeatureExtractor(model_names)
        self.ir_extractor = SuperFeatureExtractor(model_names)
        # 特征维度：每个模型3种时序特征（mean, max, std）* 模型数量 * 3个模态
        # 每个模型的特征维度 * 3（mean, max, std）* 3个模态
        self.feature_dim = self.rgb_extractor.total_feature_dim * 3 * 3
    
    def forward(self, rgb_input, depth_input, ir_input, use_multiscale=False):
        rgb_feat = self.rgb_extractor(rgb_input, use_multiscale=use_multiscale)
        depth_feat = self.depth_extractor(depth_input, use_multiscale=use_multiscale)
        ir_feat = self.ir_extractor(ir_input, use_multiscale=use_multiscale)
        # 拼接特征
        combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return combined_feat


def extract_features(model, dataloader, device, is_test_mode=False, use_multiscale=False, tta_times=1):
    """提取所有样本的特征，支持TTA和多尺度"""
    model.eval()
    all_features = []
    all_labels = []
    all_video_ids = []
    
    with torch.no_grad():
        desc = f'提取特征{"(TTA)" if tta_times > 1 else ""}'
        pbar = tqdm(dataloader, desc=desc)
        for batch in pbar:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                modalities, labels_or_ids = batch
                
                if isinstance(modalities, (tuple, list)) and len(modalities) == 3:
                    rgb_input, depth_input, ir_input = modalities
                else:
                    raise ValueError(f"意外的模态数据格式: {type(modalities)}")
                
                if is_test_mode:
                    video_ids = labels_or_ids
                    labels = None
                else:
                    labels = labels_or_ids
                    video_ids = None
            else:
                raise ValueError(f"意外的batch格式: {type(batch)}")
            
            rgb_input = rgb_input.to(device, non_blocking=True)
            depth_input = depth_input.to(device, non_blocking=True)
            ir_input = ir_input.to(device, non_blocking=True)
            
            # TTA: 多次采样取平均
            tta_features = []
            for _ in range(tta_times):
                features = model(rgb_input, depth_input, ir_input, use_multiscale=use_multiscale)
                tta_features.append(features.cpu().numpy())
            
            # 平均TTA结果
            features = np.mean(tta_features, axis=0)
            
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
    parser = argparse.ArgumentParser(description='超级无敌爆炸厉害版本 - 集成多个最强模型')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--models', type=str, default='vit_l_16,convnext_large,efficientnet_b7',
                       help='要集成的模型列表，用逗号分隔')
    parser.add_argument('--classifier', type=str, default='svm',
                       choices=['logistic', 'svm', 'rf', 'gbdt', 'ensemble'],
                       help='分类器类型（ensemble为集成多个分类器）')
    parser.add_argument('--ensemble_models', type=str, default='logistic,svm,rf',
                       help='集成学习的分类器列表（用逗号分隔），仅在classifier=ensemble时有效')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multiscale', action='store_true',
                       help='使用多尺度特征提取')
    parser.add_argument('--tta_times', type=int, default=1,
                       help='测试时增强次数（多次采样取平均）')
    parser.add_argument('--use_pca', action='store_true',
                       help='使用PCA降维')
    parser.add_argument('--pca_components', type=int, default=1000,
                       help='PCA降维后的维度')
    parser.add_argument('--use_feature_selection', action='store_true',
                       help='使用特征选择')
    parser.add_argument('--feature_selection_k', type=int, default=2000,
                       help='特征选择保留的特征数')
    parser.add_argument('--tune_params', action='store_true',
                       help='自动调优超参数（使用交叉验证）')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='交叉验证折数')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    # 解析模型列表
    model_names = [m.strip() for m in args.models.split(',')]
    print(f"\n集成模型: {model_names}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建特征提取模型
    print("\n加载预训练模型...")
    try:
        feature_extractor = MultiModalSuperExtractor(model_names).to(device)
        feature_extractor.eval()
        print(f"总特征维度: {feature_extractor.feature_dim}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("使用默认模型组合...")
        feature_extractor = MultiModalSuperExtractor(['vit_l_16', 'convnext_large']).to(device)
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
        feature_extractor, train_loader, device, is_test_mode=False,
        use_multiscale=args.use_multiscale, tta_times=1
    )
    print(f"训练特征形状: {train_features.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    
    # 特征预处理
    preprocessing_steps = [('scaler', StandardScaler())]
    
    if args.use_pca:
        print(f"\n应用PCA降维: {train_features.shape[1]} -> {args.pca_components}")
        preprocessing_steps.append(('pca', PCA(n_components=args.pca_components, random_state=42)))
    
    if args.use_feature_selection:
        print(f"\n应用特征选择: 保留前{args.feature_selection_k}个特征")
        preprocessing_steps.append(('feature_selection', SelectKBest(f_classif, k=args.feature_selection_k)))
    
    # 训练分类器（使用最强配置）
    print(f"\n训练{args.classifier}分类器...")
    
    if args.classifier == 'ensemble':
        # 集成学习：多个分类器加权平均
        ensemble_names = [m.strip() for m in args.ensemble_models.split(',')]
        print(f"集成分类器: {ensemble_names}")
        
        classifiers = []
        for name in ensemble_names:
            if name == 'logistic':
                clf = LogisticRegression(max_iter=5000, multi_class='multinomial', 
                                       solver='lbfgs', C=2.0, n_jobs=-1, random_state=42)
            elif name == 'svm':
                clf = SVC(kernel='rbf', probability=True, C=3.0, gamma='scale', 
                         decision_function_shape='ovr', max_iter=10000, random_state=42)
            elif name == 'rf':
                clf = RandomForestClassifier(n_estimators=500, max_depth=20, 
                                           n_jobs=-1, random_state=42)
            elif name == 'gbdt':
                clf = GradientBoostingClassifier(n_estimators=200, max_depth=10, 
                                                learning_rate=0.1, random_state=42)
            elif name == 'knn':
                clf = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
            else:
                print(f"警告: 未知分类器 {name}，跳过")
                continue
            
            classifiers.append((name, clf))
        
        if len(classifiers) == 0:
            raise ValueError("没有有效的分类器！")
        
        # 使用VotingClassifier进行集成
        voting_clf = VotingClassifier(estimators=classifiers, voting='soft')
        classifier = Pipeline(preprocessing_steps + [('clf', voting_clf)])
        
        # 如果启用超参数调优，使用交叉验证评估
        if args.tune_params:
            print("使用交叉验证评估集成分类器...")
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(classifier, train_features, train_labels, cv=cv, scoring='accuracy', n_jobs=-1)
            print(f"交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    elif args.classifier == 'logistic':
        if args.tune_params:
            # 超参数调优
            from sklearn.model_selection import GridSearchCV
            param_grid = {'clf__C': [0.5, 1.0, 2.0, 5.0], 'clf__solver': ['lbfgs', 'sag']}
            base_clf = Pipeline(preprocessing_steps + [('clf', LogisticRegression(max_iter=5000, multi_class='multinomial', n_jobs=-1))])
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
            classifier = GridSearchCV(base_clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        else:
            classifier = Pipeline(preprocessing_steps + [
                ('clf', LogisticRegression(max_iter=5000, multi_class='multinomial', 
                                         solver='lbfgs', C=2.0, n_jobs=-1))
            ])
    
    elif args.classifier == 'svm':
        if args.tune_params:
            from sklearn.model_selection import GridSearchCV
            param_grid = {'clf__C': [1.0, 2.0, 3.0, 5.0], 'clf__gamma': ['scale', 'auto']}
            base_clf = Pipeline(preprocessing_steps + [('clf', SVC(kernel='rbf', probability=True, decision_function_shape='ovr', max_iter=10000, random_state=42))])
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
            classifier = GridSearchCV(base_clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        else:
            classifier = Pipeline(preprocessing_steps + [
                ('clf', SVC(kernel='rbf', probability=True, C=3.0, gamma='scale', 
                           decision_function_shape='ovr', max_iter=10000))
            ])
    
    elif args.classifier == 'rf':
        classifier = Pipeline(preprocessing_steps + [
            ('clf', RandomForestClassifier(n_estimators=500, max_depth=20, 
                                         n_jobs=-1, random_state=42))
        ])
    
    else:  # gbdt
        classifier = Pipeline(preprocessing_steps + [
            ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=10, 
                                              learning_rate=0.1, random_state=42))
        ])
    
    print("开始训练分类器（这可能需要几分钟）...")
    classifier.fit(train_features, train_labels)
    
    if args.tune_params and hasattr(classifier, 'best_params_'):
        print(f"最佳超参数: {classifier.best_params_}")
        print(f"最佳交叉验证分数: {classifier.best_score_:.4f}")
    
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
        feature_extractor, test_loader, device, is_test_mode=True,
        use_multiscale=args.use_multiscale, tta_times=args.tta_times
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
    
    # 保存分类器
    classifier_path = args.output.replace('.csv', '_classifier.pkl')
    joblib.dump(classifier, classifier_path)
    print(f"\n分类器已保存到: {classifier_path}")


if __name__ == '__main__':
    main()

