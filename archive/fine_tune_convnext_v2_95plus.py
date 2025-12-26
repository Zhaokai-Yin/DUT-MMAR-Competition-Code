"""
基于92.54分，明确使用ConvNeXt V2 Large（目标95+分）
策略：确保使用ConvNeXt V2 Large，其他配置与92.54分完全一致
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

from dataset import MultiModalDataset
import torchvision.models as models

# 尝试导入MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("警告: MediaPipe未安装，将跳过骨骼点检测。安装命令: pip install mediapipe")


class ConvNeXtV2LargeExtractor(nn.Module):
    """ConvNeXt V2 Large特征提取器（尝试多种方式加载V2版本）"""
    def __init__(self):
        super(ConvNeXtV2LargeExtractor, self).__init__()
        self.model = None
        self.feature_dim = 1536
        
        # 尝试多种方式加载ConvNeXt V2 Large
        # 方式1: 使用weights参数（新版本torchvision）
        try:
            from torchvision.models import ConvNeXt_V2_Large_Weights
            self.model = models.convnext_v2_large(weights=ConvNeXt_V2_Large_Weights.IMAGENET1K_V1)
            print("  ✓ 成功加载 ConvNeXt V2 Large (使用weights参数)")
        except:
            # 方式2: 使用pretrained参数（旧版本）
            try:
                self.model = models.convnext_v2_large(pretrained=True)
                print("  ✓ 成功加载 ConvNeXt V2 Large (使用pretrained参数)")
            except:
                # 方式3: 尝试直接调用（无参数）
                try:
                    self.model = models.convnext_v2_large()
                    print("  ⚠ ConvNeXt V2 Large加载（无预训练权重）")
                except:
                    # 方式4: 降级到ConvNeXt Large
                    print("  ⚠ ConvNeXt V2 Large不可用，降级到 ConvNeXt Large")
                    try:
                        from torchvision.models import ConvNeXt_Large_Weights
                        self.model = models.convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
                        print("  ✓ 成功加载 ConvNeXt Large (使用weights参数)")
                    except:
                        try:
                            self.model = models.convnext_large(pretrained=True)
                            print("  ✓ 成功加载 ConvNeXt Large (使用pretrained参数)")
                        except:
                            raise RuntimeError("无法加载ConvNeXt模型")
        
        if self.model is None:
            raise RuntimeError("无法加载ConvNeXt模型")
        
        self.model.classifier = nn.Identity()
        self.model.eval()
    
    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        
        with torch.no_grad():
            features = self.model(x)
            features = features.view(B, T, -1)
            # 只使用mean（与92.54分配置一致）
            features = features.mean(dim=1)
        
        return features


class PoseExtractor:
    """骨骼点特征提取器（与92.54分完全一致）"""
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            self.mp_pose = None
            self.pose = None
            self.feature_dim = 0
            return
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.feature_dim = 33 * 4  # 132维
    
    def extract_pose_features(self, images):
        """从图像序列中提取骨骼点特征"""
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            if len(images.shape) == 4:
                return np.zeros(self.feature_dim)
            else:
                return np.zeros((images.shape[0], self.feature_dim))
        
        if len(images.shape) == 5:  # [B, T, H, W, C]
            B, T = images.shape[0], images.shape[1]
            all_features = []
            for b in range(B):
                video_features = []
                # 只处理关键帧（首、中、尾）
                if T > 4:
                    key_frames = [0, T//2, T-1]
                else:
                    key_frames = list(range(T))
                
                for t in key_frames:
                    img = images[b, t].copy()
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    pose_features = self._extract_single_frame(img)
                    video_features.append(pose_features)
                video_feat = np.mean(video_features, axis=0)
                all_features.append(video_feat)
            return np.array(all_features)
        else:
            return np.zeros(self.feature_dim)
    
    def _extract_single_frame(self, img):
        """从单帧提取骨骼点特征"""
        try:
            results = self.pose.process(img)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                features = []
                for landmark in landmarks:
                    features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                return np.array(features)
            else:
                return np.zeros(self.feature_dim)
        except Exception as e:
            return np.zeros(self.feature_dim)
    
    def __del__(self):
        if self.pose is not None:
            self.pose.close()


class MultiModalConvNeXtV2PoseExtractor(nn.Module):
    """多模态ConvNeXt V2 Large + 骨骼点特征提取器"""
    def __init__(self, use_pose=True):
        super(MultiModalConvNeXtV2PoseExtractor, self).__init__()
        print("\n创建ConvNeXt V2 Large提取器...")
        self.rgb_extractor = ConvNeXtV2LargeExtractor()
        self.depth_extractor = ConvNeXtV2LargeExtractor()
        self.ir_extractor = ConvNeXtV2LargeExtractor()
        self.convnext_dim = self.rgb_extractor.feature_dim * 3  # 4608
        
        if use_pose and MEDIAPIPE_AVAILABLE:
            self.pose_extractor = PoseExtractor()
            pose_dim = self.pose_extractor.feature_dim
        else:
            self.pose_extractor = None
            pose_dim = 0
        
        self.feature_dim = self.convnext_dim + pose_dim
    
    def forward(self, rgb_input, depth_input, ir_input):
        """
        提取ConvNeXt V2特征
        注意：骨骼点特征需要在CPU上单独提取
        """
        rgb_feat = self.rgb_extractor(rgb_input)
        depth_feat = self.depth_extractor(depth_input)
        ir_feat = self.ir_extractor(ir_input)
        combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return combined_feat


def extract_features_with_tta_and_pose(model, pose_extractor, dataloader, device, 
                                      is_test_mode=False, tta_times=1, use_pose=True):
    """提取特征，支持TTA和骨骼点（与92.54分完全一致）"""
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
            
            # TTA: 多次采样取平均
            tta_features = []
            for _ in range(tta_times):
                rgb_input_t = rgb_input.to(device, non_blocking=True)
                depth_input_t = depth_input.to(device, non_blocking=True)
                ir_input_t = ir_input.to(device, non_blocking=True)
                
                # 提取ConvNeXt V2特征
                convnext_features = model(rgb_input_t, depth_input_t, ir_input_t)
                tta_features.append(convnext_features.cpu().numpy())
            
            convnext_features = np.mean(tta_features, axis=0)
            
            # 提取骨骼点特征（从RGB模态）
            if use_pose and pose_extractor is not None:
                rgb_np = rgb_input.cpu().numpy()  # [B, C, T, H, W]
                rgb_np = rgb_np.transpose(0, 2, 3, 4, 1)  # [B, T, H, W, C]
                # 反归一化（ImageNet归一化）
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 1, 3)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 1, 3)
                rgb_np = rgb_np * std + mean
                rgb_np = np.clip(rgb_np, 0, 1)
                rgb_np = (rgb_np * 255).astype(np.uint8)
                # 提取骨骼点特征
                pose_features = pose_extractor.extract_pose_features(rgb_np)
                if len(pose_features.shape) == 1:
                    pose_features = pose_features.reshape(1, -1)
                # 拼接特征
                combined_features = np.concatenate([convnext_features, pose_features], axis=1)
            else:
                combined_features = convnext_features
            
            all_features.append(combined_features)
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
    parser = argparse.ArgumentParser(description='ConvNeXt V2 Large版本（目标95+分）')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--output', type=str, default='submission_convnext_v2_95plus.csv')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tta_times', type=int, default=10,
                       help='测试时增强次数（与92.54分一致）')
    parser.add_argument('--use_pose', action='store_true', default=True,
                       help='是否使用骨骼点检测')
    parser.add_argument('--classifier', type=str, default='ensemble',
                       choices=['svm', 'logistic', 'rf', 'ensemble'],
                       help='分类器类型（默认ensemble，与92.54分一致）')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    # 检查MediaPipe
    if args.use_pose and not MEDIAPIPE_AVAILABLE:
        print("警告: MediaPipe未安装，将跳过骨骼点检测")
        args.use_pose = False
    
    # 数据变换（与92.54分配置完全一致）
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建特征提取模型（明确使用ConvNeXt V2 Large）
    print("\n加载ConvNeXt V2 Large模型...")
    feature_extractor = MultiModalConvNeXtV2PoseExtractor(use_pose=args.use_pose).to(device)
    feature_extractor.eval()
    print(f"特征维度: {feature_extractor.feature_dim}")
    print(f"  - ConvNeXt V2特征: {feature_extractor.convnext_dim}维")
    if args.use_pose and feature_extractor.pose_extractor:
        print(f"  - 骨骼点特征: {feature_extractor.pose_extractor.feature_dim}维")
    
    pose_extractor = feature_extractor.pose_extractor if args.use_pose else None
    
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
    
    train_features, train_labels, _ = extract_features_with_tta_and_pose(
        feature_extractor, pose_extractor, train_loader, device, 
        is_test_mode=False, tta_times=1, use_pose=args.use_pose
    )
    print(f"训练特征形状: {train_features.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    
    # 训练分类器（与92.54分配置完全一致）
    print(f"\n训练{args.classifier}分类器...")
    
    if args.classifier == 'ensemble':
        # 集成学习（与92.54分完全一致）
        print("训练集成分类器（SVM + Logistic + RF）...")
        
        # 1. SVM
        print("  训练 SVM...")
        clf_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf', probability=True, C=3.0, gamma='scale',
                decision_function_shape='ovr', max_iter=10000, random_state=42
            ))
        ])
        clf_svm.fit(train_features, train_labels)
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
        print("  ✓ Random Forest 完成")
        
        # 交叉验证评估
        print("\n评估各个分类器...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        classifiers = [('svm', clf_svm), ('logistic', clf_lr), ('rf', clf_rf)]
        cv_scores = []
        for name, clf in classifiers:
            scores = cross_val_score(clf, train_features, train_labels, cv=cv, scoring='accuracy', n_jobs=1)
            cv_scores.append(scores.mean())
            print(f"  {name} CV准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # 加权集成（根据CV分数）
        cv_scores = np.array(cv_scores)
        weights = cv_scores / cv_scores.sum()
        print(f"\n集成权重: {dict(zip([n for n, _ in classifiers], weights))}")
        
        # 创建集成预测函数
        def ensemble_predict_proba(X):
            probas = []
            for (name, clf), weight in zip(classifiers, weights):
                proba = clf.predict_proba(X)
                probas.append(proba * weight)
            return np.sum(probas, axis=0)
        
        # 使用简单的包装类，避免pickle问题
        class EnsembleClassifier:
            def __init__(self, classifiers_list, weights_list):
                self.classifiers = classifiers_list
                self.weights = weights_list
            
            def predict_proba(self, X):
                probas = []
                for (name, clf), weight in zip(self.classifiers, self.weights):
                    proba = clf.predict_proba(X)
                    probas.append(proba * weight)
                return np.sum(probas, axis=0)
            
            def fit(self, X, y):
                return self
        
        classifier = EnsembleClassifier(classifiers, weights)
        classifier.fit(train_features, train_labels)
        print("集成分类器训练完成！")
    
    elif args.classifier == 'svm':
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf', probability=True, C=3.0, gamma='scale',
                decision_function_shape='ovr', max_iter=10000, random_state=42
            ))
        ])
        print("开始训练SVM分类器...")
        classifier.fit(train_features, train_labels)
        print("SVM分类器训练完成！")
    
    elif args.classifier == 'logistic':
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=5000, solver='lbfgs', C=2.0, n_jobs=-1, random_state=42
            ))
        ])
        print("开始训练Logistic分类器...")
        classifier.fit(train_features, train_labels)
        print("Logistic分类器训练完成！")
    
    else:  # rf
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=500, max_depth=20, n_jobs=-1, random_state=42
            ))
        ])
        print("开始训练Random Forest分类器...")
        classifier.fit(train_features, train_labels)
        print("Random Forest分类器训练完成！")
    
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
    
    test_features, _, test_video_ids = extract_features_with_tta_and_pose(
        feature_extractor, pose_extractor, test_loader, device, 
        is_test_mode=True, tta_times=args.tta_times, use_pose=args.use_pose
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
    
    # 后处理：温度缩放（与92.54分配置一致）
    predictions_proba = predictions_proba ** 0.95
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
    try:
        classifier_path = args.output.replace('.csv', '_classifier.pkl')
        if args.classifier == 'ensemble' and hasattr(classifier, 'classifiers'):
            save_data = {
                'classifiers': classifier.classifiers,
                'weights': classifier.weights,
                'type': 'ensemble'
            }
            joblib.dump(save_data, classifier_path)
            print(f"\n集成分类器已保存到: {classifier_path}")
        else:
            joblib.dump(classifier, classifier_path)
            print(f"\n分类器已保存到: {classifier_path}")
    except Exception as e:
        print(f"\n警告: 保存模型失败: {e}")
        print("（这不影响预测结果，可以忽略）")


if __name__ == '__main__':
    main()

