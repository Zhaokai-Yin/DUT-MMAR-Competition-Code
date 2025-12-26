"""
多模型预测集成版本（目标95+分，不训练）
策略：使用多个backbone分别训练分类器，然后融合它们的预测结果
这是最有可能从93.53分提升到95+分的方法！
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
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


class BackboneExtractor(nn.Module):
    """通用backbone特征提取器"""
    def __init__(self, backbone_name='convnext_large'):
        super(BackboneExtractor, self).__init__()
        self.backbone_name = backbone_name
        
        print(f"  加载 {backbone_name}...")
        try:
            if backbone_name == 'convnext_large':
                try:
                    self.model = models.convnext_v2_large(pretrained=True)
                    self.feature_dim = 1536
                    print(f"    ✓ ConvNeXt V2 Large (1536维)")
                except:
                    self.model = models.convnext_large(pretrained=True)
                    self.feature_dim = 1536
                    print(f"    ✓ ConvNeXt Large (1536维)")
            elif backbone_name == 'efficientnet_v2_l':
                self.model = models.efficientnet_v2_l(pretrained=True)
                self.feature_dim = 1280
                print(f"    ✓ EfficientNet V2 Large (1280维)")
            elif backbone_name == 'regnet_y_16gf':
                import os
                import glob
                local_weight_path = None
                for name in ['regnet_y_16gf-9e6ed7dd.pth', 'regnet_y_16gf.pth', '*regnet_y_16gf*.pth']:
                    if '*' in name:
                        matches = glob.glob(name)
                        if matches:
                            local_weight_path = matches[0]
                            break
                    elif os.path.exists(name):
                        local_weight_path = name
                        break
                
                if local_weight_path:
                    print(f"    从本地加载权重: {local_weight_path}")
                    self.model = models.regnet_y_16gf(pretrained=False)
                    state_dict = torch.load(local_weight_path, map_location='cpu')
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"    ✓ RegNet Y-16GF (3024维) - 从本地加载")
                else:
                    self.model = models.regnet_y_16gf(pretrained=True)
                    print(f"    ✓ RegNet Y-16GF (3024维) - 从网络下载")
                self.feature_dim = 3024
            else:
                raise ValueError(f"不支持的backbone: {backbone_name}")
            
            self.model.classifier = nn.Identity()
            if hasattr(self.model, 'head'):
                self.model.head = nn.Identity()
            self.model.eval()
        except Exception as e:
            print(f"    ✗ 加载失败: {e}")
            raise RuntimeError(f"无法加载backbone: {backbone_name}")
    
    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        
        with torch.no_grad():
            features = self.model(x)
            features = features.view(B, T, -1)
            features = features.mean(dim=1)
        
        return features


class PoseExtractor:
    """骨骼点特征提取器"""
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


class MultiModalBackboneExtractor(nn.Module):
    """多模态单backbone + 骨骼点特征提取器"""
    def __init__(self, backbone_name='convnext_large', use_pose=True):
        super(MultiModalBackboneExtractor, self).__init__()
        self.backbone_name = backbone_name
        
        self.rgb_extractor = BackboneExtractor(backbone_name)
        self.depth_extractor = BackboneExtractor(backbone_name)
        self.ir_extractor = BackboneExtractor(backbone_name)
        
        backbone_dim = self.rgb_extractor.feature_dim * 3  # RGB + Depth + IR
        
        if use_pose and MEDIAPIPE_AVAILABLE:
            self.pose_extractor = PoseExtractor()
            pose_dim = self.pose_extractor.feature_dim
        else:
            self.pose_extractor = None
            pose_dim = 0
        
        self.feature_dim = backbone_dim + pose_dim
    
    def forward(self, rgb_input, depth_input, ir_input):
        rgb_feat = self.rgb_extractor(rgb_input)
        depth_feat = self.depth_extractor(depth_input)
        ir_feat = self.ir_extractor(ir_input)
        combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return combined_feat


def extract_features_with_tta_and_pose(model, pose_extractor, dataloader, device, 
                                      is_test_mode=False, tta_times=1, use_pose=True):
    """提取特征，支持TTA和骨骼点"""
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
                
                backbone_features = model(rgb_input_t, depth_input_t, ir_input_t)
                tta_features.append(backbone_features.cpu().numpy())
            
            backbone_features = np.mean(tta_features, axis=0)
            
            # 提取骨骼点特征
            if use_pose and pose_extractor is not None:
                rgb_np = rgb_input.cpu().numpy()
                rgb_np = rgb_np.transpose(0, 2, 3, 4, 1)
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 1, 3)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 1, 3)
                rgb_np = rgb_np * std + mean
                rgb_np = np.clip(rgb_np, 0, 1)
                rgb_np = (rgb_np * 255).astype(np.uint8)
                pose_features = pose_extractor.extract_pose_features(rgb_np)
                if len(pose_features.shape) == 1:
                    pose_features = pose_features.reshape(1, -1)
                combined_features = np.concatenate([backbone_features, pose_features], axis=1)
            else:
                combined_features = backbone_features
            
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


def train_classifier_for_backbone(train_features, train_labels, use_feature_selection=True, 
                                  feature_selection_k=3000, use_calibration=True, skip_cv=False):
    """为单个backbone训练分类器"""
    # 特征选择
    feature_selector = None
    if use_feature_selection:
        print(f"  特征选择（保留前{feature_selection_k}个特征）...")
        feature_selector = SelectKBest(f_classif, k=min(feature_selection_k, train_features.shape[1]))
        train_features = feature_selector.fit_transform(train_features, train_labels)
        print(f"  ✓ 特征选择完成，特征维度: {train_features.shape[1]}")
    
    # 训练集成分类器
    print("  训练 SVM...")
    clf_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='rbf', probability=True, C=4.0, gamma='scale',
            decision_function_shape='ovr', max_iter=10000, random_state=42
        ))
    ])
    clf_svm.fit(train_features, train_labels)
    print("  ✓ SVM 完成")
    
    print("  训练 Logistic Regression...")
    clf_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=5000, solver='lbfgs', C=2.5, n_jobs=-1, random_state=42
        ))
    ])
    clf_lr.fit(train_features, train_labels)
    print("  ✓ Logistic Regression 完成")
    
    print("  训练 Random Forest...")
    clf_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=600, max_depth=25, n_jobs=-1, random_state=42
        ))
    ])
    clf_rf.fit(train_features, train_labels)
    print("  ✓ Random Forest 完成")
    
    classifiers = [('svm', clf_svm), ('logistic', clf_lr), ('rf', clf_rf)]
    
    # 概率校准（可能很慢，添加进度提示）
    if use_calibration:
        print("  应用概率校准（可能需要几分钟，建议跳过）...")
        calibrated_classifiers = []
        for name, clf in classifiers:
            print(f"    校准 {name}...")
            try:
                # 使用更少的折数，加快速度
                calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=2, n_jobs=1)
                calibrated_clf.fit(train_features, train_labels)
                calibrated_classifiers.append((name, calibrated_clf))
                print(f"    ✓ {name} 校准完成")
            except Exception as e:
                print(f"    ⚠ {name} 校准失败: {e}，使用原始分类器")
                calibrated_classifiers.append((name, clf))
        classifiers = calibrated_classifiers
        print("  ✓ 概率校准完成")
    
    # 交叉验证评估（可能很慢，可以跳过）
    if skip_cv:
        print("  跳过交叉验证评估（使用等权重）...")
        weights = np.array([1.0, 1.0, 1.0]) / 3.0
    else:
        print("  交叉验证评估（可能需要几分钟）...")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 减少到3折，加快速度
        cv_scores = []
        for name, clf in classifiers:
            print(f"    评估 {name}...")
            scores = cross_val_score(clf, train_features, train_labels, cv=cv, scoring='accuracy', n_jobs=1)
            cv_scores.append(scores.mean())
            print(f"    {name} CV准确率: {scores.mean():.4f}")
        
        # 加权集成
        cv_scores = np.array(cv_scores)
        weights = cv_scores / cv_scores.sum()
    
    class EnsembleClassifier:
        def __init__(self, classifiers_list, weights_list, feature_selector_obj):
            self.classifiers = classifiers_list
            self.weights = weights_list
            self.feature_selector = feature_selector_obj
        
        def predict_proba(self, X):
            if self.feature_selector is not None:
                X = self.feature_selector.transform(X)
            probas = []
            for (name, clf), weight in zip(self.classifiers, self.weights):
                proba = clf.predict_proba(X)
                probas.append(proba * weight)
            return np.sum(probas, axis=0)
    
    classifier = EnsembleClassifier(classifiers, weights, feature_selector)
    classifier.fit(train_features, train_labels)
    
    return classifier


def main():
    parser = argparse.ArgumentParser(description='多模型预测集成版本（目标95+分）')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--output', type=str, default='submission_multi_model_ensemble.csv')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--backbones', type=str, nargs='+', 
                       default=['convnext_large', 'efficientnet_v2_l'],
                       help='使用的backbone列表（推荐：convnext_large + efficientnet_v2_l）')
    parser.add_argument('--tta_times', type=int, default=10,
                       help='测试时增强次数')
    parser.add_argument('--use_pose', action='store_true', default=True,
                       help='是否使用骨骼点检测')
    parser.add_argument('--use_feature_selection', action='store_true', default=True,
                       help='是否使用特征选择')
    parser.add_argument('--feature_selection_k', type=int, default=3000,
                       help='特征选择保留的特征数')
    parser.add_argument('--use_calibration', action='store_true', default=False,
                       help='是否使用概率校准（可能很慢，默认关闭）')
    parser.add_argument('--skip_cv', action='store_true', default=False,
                       help='跳过交叉验证评估（加快速度）')
    parser.add_argument('--temperature', type=float, default=0.88,
                       help='温度缩放参数')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    # 检查MediaPipe
    if args.use_pose and not MEDIAPIPE_AVAILABLE:
        print("警告: MediaPipe未安装，将跳过骨骼点检测")
        args.use_pose = False
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 为每个backbone训练分类器并预测
    all_test_predictions = []
    backbone_weights = []
    
    print(f"\n使用 {len(args.backbones)} 个backbone进行预测集成: {args.backbones}")
    
    for backbone_name in args.backbones:
        print(f"\n{'='*60}")
        print(f"处理 backbone: {backbone_name}")
        print(f"{'='*60}")
        
        # 创建特征提取模型
        feature_extractor = MultiModalBackboneExtractor(
            backbone_name=backbone_name,
            use_pose=args.use_pose
        ).to(device)
        feature_extractor.eval()
        print(f"特征维度: {feature_extractor.feature_dim}")
        
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
        
        # 训练分类器
        print(f"\n训练分类器...")
        classifier = train_classifier_for_backbone(
            train_features, train_labels,
            use_feature_selection=args.use_feature_selection,
            feature_selection_k=args.feature_selection_k,
            use_calibration=args.use_calibration,
            skip_cv=args.skip_cv
        )
        print(f"分类器训练完成")
        
        # 提取测试集特征
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
        
        # 预测
        print("\n进行预测...")
        predictions_proba = classifier.predict_proba(test_features)
        
        # 温度缩放
        predictions_proba = predictions_proba ** args.temperature
        predictions_proba = predictions_proba / predictions_proba.sum(axis=1, keepdims=True)
        
        all_test_predictions.append(predictions_proba)
        backbone_weights.append(1.0)  # 可以后续根据CV分数调整权重
        
        # 清理显存
        del feature_extractor, classifier
        torch.cuda.empty_cache()
    
    # 融合所有backbone的预测结果
    print(f"\n{'='*60}")
    print(f"融合 {len(args.backbones)} 个backbone的预测结果...")
    print(f"{'='*60}")
    
    all_test_predictions = np.array(all_test_predictions)  # [n_backbones, n_samples, n_classes]
    backbone_weights = np.array(backbone_weights)
    backbone_weights = backbone_weights / backbone_weights.sum()
    
    print(f"Backbone权重: {dict(zip(args.backbones, backbone_weights))}")
    
    # 加权平均
    final_predictions = np.average(all_test_predictions, axis=0, weights=backbone_weights)
    
    # 如果video_ids为None，从数据集获取
    if test_video_ids is None or len(test_video_ids) == 0:
        print("从数据集中获取video_ids...")
        test_video_ids = []
        for i in range(len(test_dataset)):
            _, video_id = test_dataset[i]
            test_video_ids.append(video_id)
        test_video_ids = np.array(test_video_ids)
    
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

