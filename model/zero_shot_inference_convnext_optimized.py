"""
基于ConvNeXt Large的优化版本（91.54分基础）
目标：提升到95分+
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

from dataset import MultiModalDataset
import torchvision.models as models


class ConvNeXtLargeExtractor(nn.Module):
    """ConvNeXt Large特征提取器（91.54分基础）"""
    def __init__(self):
        super(ConvNeXtLargeExtractor, self).__init__()
        try:
            # 尝试使用ConvNeXt V2 Large（如果有）
            self.model = models.convnext_v2_large(pretrained=True)
            self.feature_dim = 1536
        except:
            # 使用ConvNeXt Large
            self.model = models.convnext_large(pretrained=True)
            self.feature_dim = 1536
        
        self.model.classifier = nn.Identity()
        self.model.eval()
    
    def forward(self, x):
        # x: [B, C, T, H, W]
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


class MultiModalConvNeXtExtractor(nn.Module):
    """多模态ConvNeXt Large特征提取器"""
    def __init__(self):
        super(MultiModalConvNeXtExtractor, self).__init__()
        self.rgb_extractor = ConvNeXtLargeExtractor()
        self.depth_extractor = ConvNeXtLargeExtractor()
        self.ir_extractor = ConvNeXtLargeExtractor()
        self.feature_dim = self.rgb_extractor.feature_dim * 3  # 4608
    
    def forward(self, rgb_input, depth_input, ir_input):
        rgb_feat = self.rgb_extractor(rgb_input)
        depth_feat = self.depth_extractor(depth_input)
        ir_feat = self.ir_extractor(ir_input)
        # 拼接特征
        combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return combined_feat


def extract_features_with_tta(model, dataloader, device, is_test_mode=False, tta_times=1):
    """提取特征，支持TTA"""
    model.eval()
    all_features = []
    all_labels = []
    all_video_ids = []
    
    with torch.no_grad():
        desc = f'提取特征{"(TTA=" + str(tta_times) + ")" if tta_times > 1 else ""}'
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
            
            # TTA: 多次采样取平均
            tta_features = []
            for _ in range(tta_times):
                rgb_input_t = rgb_input.to(device, non_blocking=True)
                depth_input_t = depth_input.to(device, non_blocking=True)
                ir_input_t = ir_input.to(device, non_blocking=True)
                
                features = model(rgb_input_t, depth_input_t, ir_input_t)
                tta_features.append(features.cpu().numpy())
            
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
    parser = argparse.ArgumentParser(description='基于ConvNeXt Large的优化版本（91.54分基础）')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--classifier', type=str, default='ensemble',
                       choices=['logistic', 'svm', 'rf', 'ensemble'],
                       help='分类器类型')
    parser.add_argument('--ensemble_models', type=str, default='logistic,svm,rf',
                       help='集成学习的分类器列表（用逗号分隔）')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--output', type=str, default='submission_convnext_optimized.csv')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tta_times', type=int, default=5,
                       help='测试时增强次数（推荐5-10）')
    parser.add_argument('--tune_params', action='store_true',
                       help='自动调优超参数')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建特征提取模型
    print("\n加载ConvNeXt Large模型...")
    feature_extractor = MultiModalConvNeXtExtractor().to(device)
    feature_extractor.eval()
    print(f"特征维度: {feature_extractor.feature_dim}")
    
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
        feature_extractor, train_loader, device, is_test_mode=False, tta_times=1
    )
    print(f"训练特征形状: {train_features.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    
    # 训练分类器
    print(f"\n训练{args.classifier}分类器...")
    
    trained_classifiers = None  # 用于保存集成分类器的子分类器
    
    if args.classifier == 'ensemble':
        # 集成学习
        ensemble_names = [m.strip() for m in args.ensemble_models.split(',')]
        print(f"集成分类器: {ensemble_names}")
        
        classifiers = []
        for name in ensemble_names:
            if name == 'logistic':
                if args.tune_params:
                    from sklearn.model_selection import GridSearchCV
                    param_grid = {'clf__C': [1.0, 2.0, 3.0, 5.0]}
                    base_clf = Pipeline([
                        ('scaler', StandardScaler()),
                        ('clf', LogisticRegression(max_iter=5000, multi_class='multinomial', solver='lbfgs', n_jobs=-1))
                    ])
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    clf = GridSearchCV(base_clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
                else:
                    clf = LogisticRegression(
                        max_iter=5000, multi_class='multinomial', 
                        solver='lbfgs', C=2.0, n_jobs=-1, random_state=42
                    )
            elif name == 'svm':
                if args.tune_params:
                    from sklearn.model_selection import GridSearchCV
                    param_grid = {'clf__C': [2.0, 3.0, 5.0, 10.0], 'clf__gamma': ['scale', 'auto']}
                    base_clf = Pipeline([
                        ('scaler', StandardScaler()),
                        ('clf', SVC(kernel='rbf', probability=True, decision_function_shape='ovr', max_iter=10000))
                    ])
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    clf = GridSearchCV(base_clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
                else:
                    clf = SVC(
                        kernel='rbf', probability=True, C=3.0, gamma='scale', 
                        decision_function_shape='ovr', max_iter=10000, random_state=42
                    )
            elif name == 'rf':
                clf = RandomForestClassifier(
                    n_estimators=500, max_depth=20, 
                    n_jobs=-1, random_state=42
                )
            else:
                print(f"警告: 未知分类器 {name}，跳过")
                continue
            
            clf_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', clf)
            ])
            classifiers.append((name, clf_pipeline))
        
        if len(classifiers) == 0:
            raise ValueError("没有有效的分类器！")
        
        # 训练每个分类器
        print("\n训练各个分类器...")
        trained_classifiers = []
        for name, clf in classifiers:
            print(f"  训练 {name}...")
            import time
            start_time = time.time()
            clf.fit(train_features, train_labels)
            elapsed = time.time() - start_time
            print(f"  {name} 训练完成，耗时: {elapsed:.2f}秒")
            
            if args.tune_params and hasattr(clf, 'best_params_'):
                print(f"  最佳参数: {clf.best_params_}")
                print(f"  最佳分数: {clf.best_score_:.4f}")
            
            trained_classifiers.append((name, clf))
        
        # 交叉验证评估
        print("\n评估集成分类器...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for name, clf in trained_classifiers:
            scores = cross_val_score(clf, train_features, train_labels, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_scores.append(scores.mean())
            print(f"  {name} CV准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # 使用加权集成（根据CV分数）
        cv_scores = np.array(cv_scores)
        weights = cv_scores / cv_scores.sum()
        print(f"\n集成权重: {dict(zip([n for n, _ in trained_classifiers], weights))}")
        
        # 创建集成预测函数
        def ensemble_predict_proba(X):
            probas = []
            for (name, clf), weight in zip(trained_classifiers, weights):
                proba = clf.predict_proba(X)
                probas.append(proba * weight)
            return np.sum(probas, axis=0)
        
        class EnsembleClassifier:
            def __init__(self, predict_func, trained_classifiers, weights):
                self.predict_proba = predict_func
                self.trained_classifiers = trained_classifiers
                self.weights = weights
                self.is_fitted = True
            
            def fit(self, X, y):
                return self
        
        classifier = EnsembleClassifier(ensemble_predict_proba, trained_classifiers, weights)
        classifier.fit(train_features, train_labels)
        print("集成分类器训练完成！")
    
    elif args.classifier == 'logistic':
        if args.tune_params:
            from sklearn.model_selection import GridSearchCV
            param_grid = {'clf__C': [1.0, 2.0, 3.0, 5.0]}
            base_clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=5000, multi_class='multinomial', solver='lbfgs', n_jobs=-1))
            ])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            classifier = GridSearchCV(base_clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        else:
            classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    max_iter=5000, multi_class='multinomial', 
                    solver='lbfgs', C=2.0, n_jobs=-1
                ))
            ])
        print("开始训练分类器...")
        classifier.fit(train_features, train_labels)
        if args.tune_params and hasattr(classifier, 'best_params_'):
            print(f"最佳参数: {classifier.best_params_}")
            print(f"最佳分数: {classifier.best_score_:.4f}")
        print("分类器训练完成！")
    
    elif args.classifier == 'svm':
        if args.tune_params:
            from sklearn.model_selection import GridSearchCV
            param_grid = {'clf__C': [2.0, 3.0, 5.0, 10.0], 'clf__gamma': ['scale', 'auto']}
            base_clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='rbf', probability=True, decision_function_shape='ovr', max_iter=10000))
            ])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            classifier = GridSearchCV(base_clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        else:
            classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(
                    kernel='rbf', probability=True, C=3.0, gamma='scale', 
                    decision_function_shape='ovr', max_iter=10000
                ))
            ])
        print("开始训练分类器（这可能需要几分钟）...")
        classifier.fit(train_features, train_labels)
        if args.tune_params and hasattr(classifier, 'best_params_'):
            print(f"最佳参数: {classifier.best_params_}")
            print(f"最佳分数: {classifier.best_score_:.4f}")
        print("分类器训练完成！")
    
    else:  # rf
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=500, max_depth=20, 
                n_jobs=-1, random_state=42
            ))
        ])
        print("开始训练分类器...")
        classifier.fit(train_features, train_labels)
        print("分类器训练完成！")
    
    # 提取测试集特征（使用TTA）
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
        tta_times=args.tta_times
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
    
    # 后处理：轻微的温度缩放（可选）
    predictions_proba = predictions_proba ** 0.9
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
    classifier_path = args.output.replace('.csv', '_classifier.pkl')
    try:
        joblib.dump(classifier, classifier_path)
        print(f"\n分类器已保存到: {classifier_path}")
    except Exception as e:
        print(f"\n警告: 保存分类器失败: {e}")
        if args.classifier == 'ensemble' and trained_classifiers is not None:
            # 对于集成分类器，尝试保存各个子分类器
            for i, (name, clf) in enumerate(trained_classifiers):
                try:
                    sub_path = args.output.replace('.csv', f'_{name}_classifier.pkl')
                    joblib.dump(clf, sub_path)
                    print(f"  已保存 {name} 分类器到: {sub_path}")
                except:
                    pass


if __name__ == '__main__':
    main()

