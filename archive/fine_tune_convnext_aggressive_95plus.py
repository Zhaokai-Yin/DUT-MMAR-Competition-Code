"""
ConvNeXt-Large激进优化版本
目标：从92.54分提升到95+分
策略：强TTA + 多次运行融合 + 更强的集成
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

from dataset import MultiModalDataset
import torchvision.models as models


class ConvNeXtLargeExtractor(nn.Module):
    """ConvNeXt Large特征提取器（92.04分基础）"""
    def __init__(self):
        super(ConvNeXtLargeExtractor, self).__init__()
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
        combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return combined_feat


def extract_features_with_tta(model, dataloader, device, is_test_mode=False, tta_times=1):
    """提取特征，支持强TTA"""
    model.eval()
    all_features = []
    all_labels = []
    all_video_ids = []
    
    import sys
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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
            
            # 强TTA：多次采样取平均
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
            
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(f"  Batch {batch_idx+1}/{total_batches} 完成")
                sys.stdout.flush()
    
    all_features = np.concatenate(all_features, axis=0)
    
    if all_labels:
        all_labels = np.array(all_labels)
        return all_features, all_labels, None
    elif all_video_ids:
        return all_features, None, np.array(all_video_ids)
    else:
        return all_features, None, None


def main():
    parser = argparse.ArgumentParser(description='ConvNeXt-Large激进优化到95+分')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--output', type=str, default='submission_convnext_aggressive_95plus.csv')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tta_times', type=int, default=20,
                       help='测试时增强次数（推荐20次）')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='多次运行取平均（推荐3次）')
    parser.add_argument('--skip_training', action='store_true',
                       help='跳过训练，只做预测（如果已有分类器）')
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
    if not args.skip_training:
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
        
        # 训练多个分类器
        print("\n训练多个分类器...")
        
        classifiers = []
        
        # 1. SVM（92.04分使用的）
        print("  训练 SVM...")
        clf_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf', probability=True, C=3.0, gamma='scale',
                decision_function_shape='ovr', max_iter=10000, random_state=42
            ))
        ])
        clf_svm.fit(train_features, train_labels)
        classifiers.append(('svm', clf_svm))
        print("  ✓ SVM 完成")
        
        # 2. Logistic Regression
        print("  训练 Logistic Regression...")
        clf_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=5000, solver='lbfgs', C=2.0, n_jobs=-1, random_state=42
            ))
        ])
        clf_lr.fit(train_features, train_labels)
        classifiers.append(('logistic', clf_lr))
        print("  ✓ Logistic Regression 完成")
        
        # 3. Random Forest
        print("  训练 Random Forest...")
        clf_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=500, max_depth=20, n_jobs=-1, random_state=42
            ))
        ])
        clf_rf.fit(train_features, train_labels)
        classifiers.append(('rf', clf_rf))
        print("  ✓ Random Forest 完成")
        
        # 4. Gradient Boosting
        print("  训练 Gradient Boosting...")
        clf_gbdt = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, max_features='sqrt', random_state=42, verbose=0
            ))
        ])
        clf_gbdt.fit(train_features, train_labels)
        classifiers.append(('gbdt', clf_gbdt))
        print("  ✓ Gradient Boosting 完成")
        
        # 交叉验证评估
        print("\n评估各个分类器...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for name, clf in classifiers:
            scores = cross_val_score(clf, train_features, train_labels, cv=cv, scoring='accuracy', n_jobs=1)
            cv_scores.append(scores.mean())
            print(f"  {name} CV准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # 保存分类器
        for name, clf in classifiers:
            clf_path = args.output.replace('.csv', f'_{name}_classifier.pkl')
            joblib.dump(clf, clf_path)
            print(f"  已保存 {name} 分类器: {clf_path}")
        
        # 保存CV分数用于加权
        cv_scores_dict = dict(zip([n for n, _ in classifiers], cv_scores))
        scores_path = args.output.replace('.csv', '_cv_scores.pkl')
        joblib.dump(cv_scores_dict, scores_path)
        print(f"  已保存CV分数: {scores_path}")
    else:
        # 加载已训练的分类器
        print("\n加载已训练的分类器...")
        classifiers = []
        cv_scores_dict = joblib.load(args.output.replace('.csv', '_cv_scores.pkl'))
        for name in ['svm', 'logistic', 'rf', 'gbdt']:
            clf_path = args.output.replace('.csv', f'_{name}_classifier.pkl')
            if os.path.exists(clf_path):
                clf = joblib.load(clf_path)
                classifiers.append((name, clf))
                print(f"  ✓ 加载 {name} 分类器")
        
        cv_scores = [cv_scores_dict[n] for n, _ in classifiers]
    
    # 加权集成（根据CV分数）
    cv_scores = np.array(cv_scores)
    weights = cv_scores / cv_scores.sum()
    print(f"\n集成权重: {dict(zip([n for n, _ in classifiers], weights))}")
    
    # 提取测试集特征并多次运行
    print(f"\n提取测试集特征（TTA={args.tta_times}，运行{args.num_runs}次）...")
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
    
    # 多次运行取平均
    all_predictions = []
    for run_idx in range(args.num_runs):
        print(f"\n运行 {run_idx+1}/{args.num_runs}...")
        test_features, _, test_video_ids = extract_features_with_tta(
            feature_extractor, test_loader, device, is_test_mode=True,
            tta_times=args.tta_times
        )
        print(f"测试特征形状: {test_features.shape}")
        
        # 集成预测
        probas = []
        for (name, clf), weight in zip(classifiers, weights):
            proba = clf.predict_proba(test_features)
            probas.append(proba * weight)
        predictions_proba = np.sum(probas, axis=0)
        
        all_predictions.append(predictions_proba)
    
    # 平均多次运行的预测
    final_predictions = np.mean(all_predictions, axis=0)
    print(f"\n融合{args.num_runs}次运行的预测结果...")
    
    # 如果video_ids为None，从数据集获取
    if test_video_ids is None or len(test_video_ids) == 0:
        print("从数据集中获取video_ids...")
        test_video_ids = []
        for i in range(len(test_dataset)):
            _, video_id = test_dataset[i]
            test_video_ids.append(video_id)
        test_video_ids = np.array(test_video_ids)
    
    # 后处理：温度缩放
    final_predictions = final_predictions ** 0.95
    final_predictions = final_predictions / final_predictions.sum(axis=1, keepdims=True)
    
    # 获取Top-5预测
    top5_indices = np.argsort(final_predictions, axis=1)[:, -5:][:, ::-1]
    
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


if __name__ == '__main__':
    main()




