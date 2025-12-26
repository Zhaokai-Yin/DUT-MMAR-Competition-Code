"""
超强集成推理 - 集成多个最强backbone + TTA
目标：指标突破92+
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
import joblib

from dataset import MultiModalDataset
from model_ultra import MultiModalUltra


class UltraFeatureExtractor(nn.Module):
    """超强backbone特征提取器"""
    def __init__(self, backbone='convnext', model_size='base'):
        super(UltraFeatureExtractor, self).__init__()
        self.backbone = backbone
        self.model_size = model_size
        
        self.model = MultiModalUltra(
            num_class=20,
            num_segments=8,
            backbone=backbone,
            model_size=model_size,
            fusion_method='late',
            dropout=0.0
        )
        self.model.eval()
    
    def forward(self, rgb_input, depth_input, ir_input):
        """提取多模态特征"""
        B, C, T, H, W = rgb_input.size()
        
        rgb_flat = rgb_input.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        rgb_feat = self._extract_single_feature(self.model.rgb_model, rgb_flat)
        rgb_feat = rgb_feat.view(B, T, -1).mean(dim=1)
        
        depth_flat = depth_input.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        depth_feat = self._extract_single_feature(self.model.depth_model, depth_flat)
        depth_feat = depth_feat.view(B, T, -1).mean(dim=1)
        
        ir_flat = ir_input.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        ir_feat = self._extract_single_feature(self.model.ir_model, ir_flat)
        ir_feat = ir_feat.view(B, T, -1).mean(dim=1)
        
        features = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return features
    
    def _extract_single_feature(self, model, x):
        return model.base_model(x)


def extract_features_with_tta(model, dataloader, device, use_tta=False, is_test=False):
    """提取特征（带TTA）"""
    model.eval()
    all_features = []
    all_labels = []
    all_video_ids = []
    
    with torch.no_grad():
        desc = '[Extract Features with TTA]' if use_tta else '[Extract Features]'
        pbar = tqdm(dataloader, desc=desc)
        for batch_data in pbar:
            (rgb_input, depth_input, ir_input), second_item = batch_data
            
            if is_test:
                video_ids = second_item
                labels = None
            else:
                labels = second_item
                video_ids = None
            
            # TTA: 对输入进行水平翻转
            if use_tta:
                tta_features = []
                
                # 原始
                rgb_orig = rgb_input.to(device, non_blocking=True)
                depth_orig = depth_input.to(device, non_blocking=True)
                ir_orig = ir_input.to(device, non_blocking=True)
                features_orig = model(rgb_orig, depth_orig, ir_orig)
                tta_features.append(features_orig.cpu().numpy())
                
                # 水平翻转
                rgb_flip = torch.flip(rgb_input, dims=[-1]).to(device, non_blocking=True)
                depth_flip = torch.flip(depth_input, dims=[-1]).to(device, non_blocking=True)
                ir_flip = torch.flip(ir_input, dims=[-1]).to(device, non_blocking=True)
                features_flip = model(rgb_flip, depth_flip, ir_flip)
                tta_features.append(features_flip.cpu().numpy())
                
                # 平均TTA结果
                features = np.mean(tta_features, axis=0)
            else:
                rgb_input = rgb_input.to(device, non_blocking=True)
                depth_input = depth_input.to(device, non_blocking=True)
                ir_input = ir_input.to(device, non_blocking=True)
                
                features = model(rgb_input, depth_input, ir_input)
                features = features.cpu().numpy()
            
            all_features.append(features)
            
            if labels is not None:
                all_labels.append(labels.numpy())
            
            if video_ids is not None:
                if isinstance(video_ids, torch.Tensor):
                    if video_ids.dim() == 0:
                        all_video_ids.append(video_ids.item())
                    else:
                        all_video_ids.extend(video_ids.cpu().numpy().tolist())
                else:
                    all_video_ids.extend(video_ids)
    
    all_features = np.concatenate(all_features, axis=0)
    
    if all_labels:
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_labels = None
    
    return all_features, all_labels, all_video_ids


def extract_features(model, dataloader, device, is_test=False):
    """提取特征（无TTA）"""
    return extract_features_with_tta(model, dataloader, device, use_tta=False)


def train_ensemble_classifiers(X_train, y_train, use_gb=True):
    """训练集成分类器"""
    classifiers = {}
    
    # Logistic Regression
    print("训练 Logistic Regression...")
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=3000,
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0,
            random_state=42
        ))
    ])
    lr.fit(X_train, y_train)
    classifiers['lr'] = lr
    
    # SVM with RBF kernel
    print("训练 SVM (RBF)...")
    svm_rbf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            kernel='rbf',
            probability=True,
            C=1.0,
            gamma='scale',
            random_state=42
        ))
    ])
    svm_rbf.fit(X_train, y_train)
    classifiers['svm_rbf'] = svm_rbf
    
    # SVM with polynomial kernel
    print("训练 SVM (Poly)...")
    svm_poly = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            kernel='poly',
            degree=3,
            probability=True,
            C=1.0,
            random_state=42
        ))
    ])
    svm_poly.fit(X_train, y_train)
    classifiers['svm_poly'] = svm_poly
    
    # Random Forest
    print("训练 Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    classifiers['rf'] = rf
    
    # Gradient Boosting（可选，较慢）
    if use_gb:
        print("训练 Gradient Boosting...")
        print("  (这可能需要几分钟，请耐心等待...)")
        gb = GradientBoostingClassifier(
            n_estimators=100,  # 从200减少到100，加快训练
            learning_rate=0.1,
            max_depth=8,  # 从10减少到8，加快训练
            subsample=0.8,  # 添加子采样，加快训练
            random_state=42,
            verbose=1  # 显示进度
        )
        gb.fit(X_train, y_train)
        classifiers['gb'] = gb
    else:
        print("跳过 Gradient Boosting（使用 --skip_gb 跳过）")
    
    return classifiers


def predict_ensemble_weighted(classifiers, X_test, weights=None):
    """加权集成预测"""
    if weights is None:
        # 默认权重：给更强的分类器更高权重
        # 根据可用的分类器动态调整权重
        base_weights = {
            'lr': 0.15,
            'svm_rbf': 0.25,
            'svm_poly': 0.20,
            'rf': 0.25,
            'gb': 0.15
        }
        # 只使用存在的分类器
        weights = {k: v for k, v in base_weights.items() if k in classifiers}
        # 归一化权重
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
    
    all_probs = []
    weight_list = []
    
    for name, clf in classifiers.items():
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_test)
        else:
            probs = clf.decision_function(X_test)
            probs = torch.softmax(torch.tensor(probs), dim=1).numpy()
        
        weight = weights.get(name, 1.0 / len(classifiers))
        all_probs.append(probs * weight)
        weight_list.append(weight)
    
    # 加权平均
    avg_probs = np.sum(all_probs, axis=0) / sum(weight_list)
    
    # Top-5预测
    top5_indices = np.argsort(avg_probs, axis=1)[:, -5:][:, ::-1]
    
    return top5_indices


def main():
    parser = argparse.ArgumentParser(description='超强集成推理 - 多个backbone + TTA')
    parser.add_argument('--data_root', type=str, default='MMAR/train_500')
    parser.add_argument('--test_root', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_train', type=str, 
                       default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--video_list_test', type=str,
                       default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--backbones', type=str, nargs='+',
                       default=['convnext', 'resnext', 'efficientnet_v2'],
                       help='要集成的backbone列表')
    parser.add_argument('--model_sizes', type=str, nargs='+',
                       default=['large', '101', 'l'],
                       help='对应的模型大小')
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--use_tta', action='store_true',
                       help='使用测试时增强')
    parser.add_argument('--skip_gb', action='store_true',
                       help='跳过Gradient Boosting（加快训练）')
    parser.add_argument('--output', type=str, default='submission_ensemble_ultra.csv')
    parser.add_argument('--save_classifier', type=str, default='')
    parser.add_argument('--load_classifier', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'集成Backbone: {args.backbones}')
    print(f'模型大小: {args.model_sizes}')
    print(f'使用TTA: {args.use_tta}')
    
    # 基础变换
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # TTA标志
    if args.use_tta:
        print("启用测试时增强（TTA）- 水平翻转...")
    
    # 创建数据集
    print("加载训练数据...")
    train_dataset = MultiModalDataset(
        args.data_root, args.video_list_train,
        num_segments=args.num_segments,
        transform=base_transform,
        random_shift=False,
        test_mode=False
    )
    
    print("加载测试数据...")
    test_dataset = MultiModalDataset(
        args.test_root, args.video_list_test,
        num_segments=args.num_segments,
        transform=base_transform,
        random_shift=False,
        test_mode=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 提取所有backbone的特征
    all_train_features = []
    all_test_features = []
    
    for backbone, model_size in zip(args.backbones, args.model_sizes):
        print(f"\n处理Backbone: {backbone} ({model_size})")
        
        feature_extractor = UltraFeatureExtractor(
            backbone=backbone,
            model_size=model_size
        ).to(device)
        
        # 提取训练特征
        print("提取训练特征...")
        X_train, y_train, _ = extract_features(feature_extractor, train_loader, device, is_test=False)
        print(f"训练特征形状: {X_train.shape}")
        
        # 提取测试特征
        print("提取测试特征...")
        X_test, _, test_video_ids = extract_features_with_tta(
            feature_extractor, test_loader, device, use_tta=args.use_tta, is_test=True
        )
        print(f"测试特征形状: {X_test.shape}")
        
        all_train_features.append(X_train)
        all_test_features.append(X_test)
        
        # 清理GPU内存
        del feature_extractor
        torch.cuda.empty_cache()
    
    # 拼接所有backbone的特征
    print("\n融合所有backbone特征...")
    X_train_ensemble = np.concatenate(all_train_features, axis=1)
    X_test_ensemble = np.concatenate(all_test_features, axis=1)
    print(f"集成训练特征形状: {X_train_ensemble.shape}")
    print(f"集成测试特征形状: {X_test_ensemble.shape}")
    
    # 训练或加载分类器
    if args.load_classifier and os.path.exists(args.load_classifier):
        print(f"加载分类器: {args.load_classifier}")
        classifiers = joblib.load(args.load_classifier)
    else:
        print("训练集成分类器...")
        classifiers = train_ensemble_classifiers(X_train_ensemble, y_train, use_gb=not args.skip_gb)
        
        if args.save_classifier:
            print(f"保存分类器: {args.save_classifier}")
            joblib.dump(classifiers, args.save_classifier)
    
    # 预测
    print("集成预测...")
    predictions = predict_ensemble_weighted(classifiers, X_test_ensemble)
    
    # 格式化结果
    prediction_strings = []
    for pred in predictions:
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

