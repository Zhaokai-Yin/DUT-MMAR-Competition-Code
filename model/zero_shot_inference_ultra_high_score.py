"""
超高分优化版本 - 目标95分+
使用最激进的优化策略：多模型集成、强TTA、特征工程、后处理等
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

from dataset import MultiModalDataset
import torchvision.models.video as video_models
import torchvision.models as models

# 尝试导入transformers
try:
    from transformers import AutoModel, AutoConfig, AutoImageProcessor
    HAS_TRANSFORMERS = True
except:
    HAS_TRANSFORMERS = False

try:
    import easydict
    HAS_EASYDICT = True
except:
    HAS_EASYDICT = False


class MultiModelEnsembleExtractor(nn.Module):
    """多模型集成特征提取器 - 集成多个最强模型"""
    def __init__(self, model_list=['r2plus1d_18', 'r3d_18', 'mc3_18']):
        super(MultiModelEnsembleExtractor, self).__init__()
        self.extractors = nn.ModuleList()
        self.feature_dims = []
        
        for model_name in model_list:
            extractor = self._create_extractor(model_name)
            self.extractors.append(extractor)
            self.feature_dims.append(extractor.feature_dim)
        
        self.total_feature_dim = sum(self.feature_dims)
    
    def _create_extractor(self, model_name):
        """创建单个特征提取器"""
        extractor = nn.Module()
        
        if model_name == 'r2plus1d_18':
            model = video_models.r2plus1d_18(pretrained=True)
            model.fc = nn.Identity()
            extractor.model = model
            extractor.feature_dim = 512
        elif model_name == 'r3d_18':
            model = video_models.r3d_18(pretrained=True)
            model.fc = nn.Identity()
            extractor.model = model
            extractor.feature_dim = 512
        elif model_name == 'mc3_18':
            model = video_models.mc3_18(pretrained=True)
            model.fc = nn.Identity()
            extractor.model = model
            extractor.feature_dim = 512
        elif model_name == 'efficientnet_b7' and HAS_TRANSFORMERS:
            # 图像模型作为补充
            model = models.efficientnet_b7(pretrained=True)
            model.classifier = nn.Identity()
            extractor.model = model
            extractor.feature_dim = 2560
            extractor.is_image_model = True
        else:
            # 默认
            model = video_models.r2plus1d_18(pretrained=True)
            model.fc = nn.Identity()
            extractor.model = model
            extractor.feature_dim = 512
        
        extractor.model.eval()
        return extractor
    
    def forward(self, x, use_multiscale=False):
        """
        x: [B, C, T, H, W]
        """
        B, C, T, H, W = x.size()
        all_features = []
        
        with torch.no_grad():
            for extractor in self.extractors:
                if hasattr(extractor, 'is_image_model') and extractor.is_image_model:
                    # 图像模型：对每帧提取特征后平均
                    x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
                    feat = extractor.model(x_flat)
                    feat = feat.view(B, T, -1).mean(dim=1)  # 时序平均
                else:
                    # 视频模型
                    x_video = x.clone()
                    # 确保帧数是16的倍数
                    if T < 16:
                        repeat_times = (16 + T - 1) // T
                        x_video = x_video.repeat(1, 1, repeat_times, 1, 1)[:, :, :16, :, :]
                    elif T > 16:
                        indices = torch.linspace(0, T - 1, 16).long()
                        x_video = x_video[:, :, indices, :, :]
                    
                    feat = extractor.model(x_video)
                
                # 时序聚合：只使用mean（更稳定，避免过拟合）
                if len(feat.shape) == 2:
                    feat_combined = feat
                else:
                    feat_combined = feat.mean(dim=1)  # 只使用mean，不用max和std
                
                all_features.append(feat_combined)
        
        # 拼接所有模型特征
        combined = torch.cat(all_features, dim=1)
        return combined


class MultiModalUltraExtractor(nn.Module):
    """多模态超强特征提取器"""
    def __init__(self, model_list=['r2plus1d_18', 'r3d_18', 'mc3_18']):
        super(MultiModalUltraExtractor, self).__init__()
        self.rgb_extractor = MultiModelEnsembleExtractor(model_list)
        self.depth_extractor = MultiModelEnsembleExtractor(model_list)
        self.ir_extractor = MultiModelEnsembleExtractor(model_list)
        # 特征维度：每个模型 * 模型数量 * 3个模态（只使用mean，不用max/std）
        self.feature_dim = self.rgb_extractor.total_feature_dim * 3
    
    def forward(self, rgb_input, depth_input, ir_input, use_multiscale=False):
        rgb_feat = self.rgb_extractor(rgb_input, use_multiscale)
        depth_feat = self.depth_extractor(depth_input, use_multiscale)
        ir_feat = self.ir_extractor(ir_input, use_multiscale)
        # 拼接特征
        combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return combined_feat


def extract_features_with_tta(model, dataloader, device, is_test_mode=False, tta_times=5, use_multiscale=False):
    """提取特征，支持强TTA和多尺度"""
    model.eval()
    all_features = []
    all_labels = []
    all_video_ids = []
    
    # 多尺度变换
    scales = [224, 256, 192] if use_multiscale else [224]
    
    with torch.no_grad():
        desc = f'提取特征(TTA={tta_times}, Scales={len(scales)})'
        pbar = tqdm(dataloader, desc=desc)
        for batch in pbar:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                modalities, labels_or_ids = batch
                
                if isinstance(modalities, (tuple, list)) and len(modalities) == 3:
                    rgb_input, depth_input, ir_input = modalities
                else:
                    raise ValueError(f"意外的模态数据格式")
                
                if is_test_mode:
                    video_ids = labels_or_ids
                    labels = None
                else:
                    labels = labels_or_ids
                    video_ids = None
            else:
                raise ValueError(f"意外的batch格式")
            
            # TTA: 多次采样 + 多尺度
            tta_features = []
            for tta_idx in range(tta_times):
                for scale in scales:
                    # 调整尺寸
                    if scale != 224:
                        rgb_scaled = F.interpolate(
                            rgb_input.view(-1, rgb_input.shape[1], rgb_input.shape[3], rgb_input.shape[4]),
                            size=(scale, scale), mode='bilinear', align_corners=False
                        ).view(rgb_input.shape[0], rgb_input.shape[1], rgb_input.shape[2], scale, scale)
                        depth_scaled = F.interpolate(
                            depth_input.view(-1, depth_input.shape[1], depth_input.shape[3], depth_input.shape[4]),
                            size=(scale, scale), mode='bilinear', align_corners=False
                        ).view(depth_input.shape[0], depth_input.shape[1], depth_input.shape[2], scale, scale)
                        ir_scaled = F.interpolate(
                            ir_input.view(-1, ir_input.shape[1], ir_input.shape[3], ir_input.shape[4]),
                            size=(scale, scale), mode='bilinear', align_corners=False
                        ).view(ir_input.shape[0], ir_input.shape[1], ir_input.shape[2], scale, scale)
                    else:
                        rgb_scaled = rgb_input
                        depth_scaled = depth_input
                        ir_scaled = ir_input
                    
                    rgb_scaled = rgb_scaled.to(device, non_blocking=True)
                    depth_scaled = depth_scaled.to(device, non_blocking=True)
                    ir_scaled = ir_scaled.to(device, non_blocking=True)
                    
                    features = model(rgb_scaled, depth_scaled, ir_scaled, use_multiscale=False)
                    tta_features.append(features.cpu().numpy())
            
            # 平均所有TTA结果
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


def create_advanced_classifier(classifier_type, use_calibration=True, use_stacking=False):
    """创建高级分类器"""
    base_classifiers = []
    
    # 基础分类器
    if 'logistic' in classifier_type or classifier_type == 'ensemble':
        base_classifiers.append(('logistic', LogisticRegression(
            max_iter=10000, multi_class='multinomial', 
            solver='lbfgs', C=3.0, n_jobs=-1, random_state=42
        )))
    
    if 'svm' in classifier_type or classifier_type == 'ensemble':
        base_classifiers.append(('svm', SVC(
            kernel='rbf', probability=True, C=5.0, gamma='scale', 
            decision_function_shape='ovr', max_iter=20000, random_state=42
        )))
    
    if 'rf' in classifier_type or classifier_type == 'ensemble':
        base_classifiers.append(('rf', RandomForestClassifier(
            n_estimators=1000, max_depth=25, min_samples_split=2,
            min_samples_leaf=1, n_jobs=-1, random_state=42
        )))
    
    if 'gbdt' in classifier_type or classifier_type == 'ensemble':
        # 优化GBDT参数以加快训练速度
        base_classifiers.append(('gbdt', GradientBoostingClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.1,  # 减少树的数量和深度
            subsample=0.8, max_features='sqrt',  # 使用sqrt特征采样加快速度
            random_state=42, verbose=1  # 显示训练进度
        )))
    
    if 'knn' in classifier_type or classifier_type == 'ensemble':
        base_classifiers.append(('knn', KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='minkowski', p=2, n_jobs=-1
        )))
    
    if len(base_classifiers) == 0:
        # 默认
        base_classifiers.append(('logistic', LogisticRegression(
            max_iter=10000, multi_class='multinomial', 
            solver='lbfgs', C=2.0, n_jobs=-1, random_state=42
        )))
    
    if use_stacking and len(base_classifiers) > 1:
        # 使用Stacking
        meta_classifier = LogisticRegression(max_iter=5000, random_state=42)
        classifier = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=meta_classifier,
            cv=5,
            n_jobs=-1
        )
    else:
        # 使用Voting
        classifier = VotingClassifier(
            estimators=base_classifiers,
            voting='soft',
            n_jobs=-1
        )
    
    if use_calibration:
        # 校准分类器
        classifier = CalibratedClassifierCV(classifier, method='isotonic', cv=3)
    
    return classifier


def main():
    parser = argparse.ArgumentParser(description='超高分优化版本 - 目标95分+')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--models', type=str, default='r2plus1d_18,r3d_18,mc3_18',
                       help='要集成的模型列表，用逗号分隔')
    parser.add_argument('--classifier', type=str, default='ensemble',
                       choices=['logistic', 'svm', 'rf', 'gbdt', 'ensemble', 'stacking'],
                       help='分类器类型')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_segments', type=int, default=16)
    parser.add_argument('--output', type=str, default='submission_ultra_high.csv')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tta_times', type=int, default=5,
                       help='测试时增强次数（推荐5-10）')
    parser.add_argument('--use_multiscale', action='store_true',
                       help='使用多尺度特征提取')
    parser.add_argument('--use_pca', action='store_true',
                       help='使用PCA降维')
    parser.add_argument('--pca_components', type=int, default=2000,
                       help='PCA降维后的维度')
    parser.add_argument('--use_feature_selection', action='store_true',
                       help='使用特征选择')
    parser.add_argument('--feature_selection_k', type=int, default=3000,
                       help='特征选择保留的特征数')
    parser.add_argument('--use_calibration', action='store_true', default=True,
                       help='使用概率校准')
    parser.add_argument('--use_stacking', action='store_true',
                       help='使用Stacking集成（比Voting更强）')
    parser.add_argument('--ensemble_predictions', action='store_true',
                       help='集成多个分类器的预测结果')
    parser.add_argument('--skip_gbdt', action='store_true',
                       help='跳过GBDT分类器（如果训练太慢）')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    # 解析模型列表
    model_list = [m.strip() for m in args.models.split(',')]
    print(f"\n集成模型: {model_list}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建特征提取模型
    print("\n加载多模型集成特征提取器...")
    feature_extractor = MultiModalUltraExtractor(model_list).to(device)
    feature_extractor.eval()
    print(f"总特征维度: {feature_extractor.feature_dim}")
    
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
    
    train_features, train_labels, _ = extract_features_with_tta(
        feature_extractor, train_loader, device, is_test_mode=False,
        tta_times=1, use_multiscale=False
    )
    print(f"训练特征形状: {train_features.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    
    # 特征预处理
    preprocessing_steps = []
    
    # 使用StandardScaler（更稳定，RobustScaler可能过于激进）
    preprocessing_steps.append(('scaler', StandardScaler()))
    
    if args.use_pca:
        print(f"\n应用PCA降维: {train_features.shape[1]} -> {args.pca_components}")
        preprocessing_steps.append(('pca', PCA(n_components=args.pca_components, random_state=42)))
    
    if args.use_feature_selection:
        print(f"\n应用特征选择: 保留前{args.feature_selection_k}个特征")
        preprocessing_steps.append(('feature_selection', SelectKBest(mutual_info_classif, k=args.feature_selection_k)))
    
    # 训练分类器
    print(f"\n训练{args.classifier}分类器...")
    
    if args.ensemble_predictions:
        # 训练多个分类器并集成预测（手动集成）
        classifiers = []
        clf_names = ['logistic', 'svm', 'rf']
        if not args.skip_gbdt:
            clf_names.append('gbdt')
        
        for i, clf_name in enumerate(clf_names, 1):
            print(f"\n[{i}/{len(clf_names)}] 训练 {clf_name} 分类器...")
            clf = create_advanced_classifier(clf_name, use_calibration=args.use_calibration, use_stacking=False)
            clf = Pipeline(preprocessing_steps + [('clf', clf)])
            
            # 对于GBDT，使用PCA降维以加快训练
            if clf_name == 'gbdt' and args.use_pca:
                print("  GBDT训练较慢，已使用PCA降维加速...")
            
            import time
            start_time = time.time()
            clf.fit(train_features, train_labels)
            elapsed_time = time.time() - start_time
            print(f"  {clf_name} 分类器训练完成，耗时: {elapsed_time:.2f}秒")
            classifiers.append((clf_name, clf))
        
        # 保存classifiers列表供后续使用
        # 创建一个包装类来保存classifiers
        class EnsembleClassifier:
            def __init__(self, classifiers_list):
                self.classifiers = classifiers_list
                self.is_fitted = True
            
            def fit(self, X, y):
                # 已经训练过了
                return self
            
            def predict_proba(self, X):
                probas = []
                for name, clf in self.classifiers:
                    proba = clf.predict_proba(X)
                    probas.append(proba)
                # 加权平均（根据分类器数量动态调整权重）
                n_clfs = len(self.classifiers)
                if n_clfs == 4:
                    weights = np.array([0.25, 0.3, 0.25, 0.2])  # logistic, svm, rf, gbdt
                elif n_clfs == 3:
                    weights = np.array([0.3, 0.4, 0.3])  # logistic, svm, rf (SVM权重最高)
                else:
                    weights = np.ones(n_clfs) / n_clfs  # 等权重
                ensemble_proba = np.average(probas, axis=0, weights=weights)
                return ensemble_proba
        
        classifier = EnsembleClassifier(classifiers)
        classifier.fit(train_features, train_labels)
        print("集成分类器训练完成！")
    elif args.classifier == 'stacking' or args.use_stacking:
        classifier = create_advanced_classifier('ensemble', use_calibration=args.use_calibration, use_stacking=True)
        classifier = Pipeline(preprocessing_steps + [('clf', classifier)])
        print("开始训练Stacking分类器（这可能需要几分钟）...")
        classifier.fit(train_features, train_labels)
        print("分类器训练完成！")
    elif args.classifier == 'ensemble':
        # 使用Voting集成
        classifier = create_advanced_classifier('ensemble', use_calibration=args.use_calibration, use_stacking=False)
        classifier = Pipeline(preprocessing_steps + [('clf', classifier)])
        print("开始训练Ensemble分类器（这可能需要几分钟）...")
        classifier.fit(train_features, train_labels)
        print("分类器训练完成！")
    else:
        classifier = create_advanced_classifier(args.classifier, use_calibration=args.use_calibration, use_stacking=False)
        classifier = Pipeline(preprocessing_steps + [('clf', classifier)])
        print("开始训练分类器（这可能需要几分钟）...")
        classifier.fit(train_features, train_labels)
        print("分类器训练完成！")
    
    # 交叉验证评估
    if not args.ensemble_predictions:
        print("\n进行交叉验证评估...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(classifier, train_features, train_labels, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # 提取测试集特征（使用强TTA）
    print(f"\n提取测试集特征（TTA={args.tta_times}）...")
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
    
    test_features, _, test_video_ids = extract_features_with_tta(
        feature_extractor, test_loader, device, is_test_mode=True,
        tta_times=args.tta_times, use_multiscale=args.use_multiscale
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
    
    # 后处理：平滑预测概率
    predictions_proba = predictions_proba ** 0.8  # 温度缩放
    predictions_proba = predictions_proba / predictions_proba.sum(axis=1, keepdims=True)
    
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
    if not args.ensemble_predictions:
        classifier_path = args.output.replace('.csv', '_classifier.pkl')
        joblib.dump(classifier, classifier_path)
        print(f"\n分类器已保存到: {classifier_path}")


if __name__ == '__main__':
    main()


