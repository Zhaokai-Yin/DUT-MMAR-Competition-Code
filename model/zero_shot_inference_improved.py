"""
零样本/少样本推理 - 改进版（基于原始成功版本）
在保持稳定性的基础上添加多种提升方法
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
from sklearn.model_selection import cross_val_score
import joblib

from dataset import MultiModalDataset
import torchvision.models.video as video_models


class PretrainedFeatureExtractor(nn.Module):
    """使用预训练视频模型提取特征"""
    def __init__(self, model_name='r2plus1d_18'):
        super(PretrainedFeatureExtractor, self).__init__()
        self.model_name = model_name
        
        if model_name == 'r2plus1d_18':
            self.model = video_models.r2plus1d_18(pretrained=True)
            self.model.fc = nn.Identity()
            self.feature_dim = 512
            self.required_frames = 16
        elif model_name == 'r3d_18':
            self.model = video_models.r3d_18(pretrained=True)
            self.model.fc = nn.Identity()
            self.feature_dim = 512
            self.required_frames = 16
        elif model_name == 'mc3_18':
            self.model = video_models.mc3_18(pretrained=True)
            self.model.fc = nn.Identity()
            self.feature_dim = 512
            self.required_frames = 16
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model.eval()
    
    def forward(self, x):
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
        return features


class MultiModalFeatureExtractor(nn.Module):
    """多模态特征提取器"""
    def __init__(self, model_name='r2plus1d_18'):
        super(MultiModalFeatureExtractor, self).__init__()
        self.rgb_extractor = PretrainedFeatureExtractor(model_name)
        self.depth_extractor = PretrainedFeatureExtractor(model_name)
        self.ir_extractor = PretrainedFeatureExtractor(model_name)
        self.feature_dim = self.rgb_extractor.feature_dim * 3
    
    def forward(self, rgb_input, depth_input, ir_input):
        rgb_feat = self.rgb_extractor(rgb_input)
        depth_feat = self.depth_extractor(depth_input)
        ir_feat = self.ir_extractor(ir_input)
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
            (rgb_input, depth_input, ir_input), targets = batch
            
            rgb_input = rgb_input.to(device, non_blocking=True)
            depth_input = depth_input.to(device, non_blocking=True)
            ir_input = ir_input.to(device, non_blocking=True)
            
            features = model(rgb_input, depth_input, ir_input)
            features = features.cpu().numpy()
            
            all_features.append(features)
            
            if is_test_mode:
                all_video_ids.extend(targets.numpy())
            else:
                all_labels.extend(targets.numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    
    if is_test_mode:
        return all_features, None, np.array(all_video_ids) if all_video_ids else None
    else:
        return all_features, np.array(all_labels) if all_labels else None, None


def create_classifier(classifier_type, C=None, n_neighbors=None):
    """创建分类器，支持自定义超参数"""
    steps = [('scaler', StandardScaler())]
    
    if classifier_type == 'logistic':
        if C is None:
            C = 1.0
        clf = LogisticRegression(
            max_iter=1000, 
            multi_class='multinomial', 
            solver='lbfgs',
            C=C
        )
    elif classifier_type == 'svm':
        if C is None:
            C = 1.0
        clf = SVC(kernel='rbf', probability=True, C=C)
    elif classifier_type == 'knn':
        if n_neighbors is None:
            n_neighbors = 5
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    else:
        raise ValueError(f"不支持的分类器: {classifier_type}")
    
    steps.append(('clf', clf))
    return Pipeline(steps)


def simple_ensemble_predict(classifiers, features):
    """简单集成：平均多个分类器的预测概率"""
    all_proba = []
    for clf in classifiers:
        proba = clf.predict_proba(features)
        all_proba.append(proba)
    
    # 简单平均
    ensemble_proba = np.mean(all_proba, axis=0)
    return ensemble_proba


def main():
    parser = argparse.ArgumentParser(description='零样本/少样本推理 - 改进版')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--model_name', type=str, default='r2plus1d_18',
                       choices=['r2plus1d_18', 'r3d_18', 'mc3_18'],
                       help='预训练模型')
    parser.add_argument('--classifier', type=str, default='logistic',
                       choices=['logistic', 'svm', 'knn', 'ensemble'],
                       help='分类器类型')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_segments', type=int, default=16,
                       help='视频分段数量')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--gpu', type=int, default=0)
    
    # 改进选项
    parser.add_argument('--tune_params', action='store_true',
                       help='自动调优超参数')
    parser.add_argument('--ensemble_models', type=str, nargs='+',
                       default=['logistic', 'svm'],
                       help='集成学习的分类器列表（当classifier=ensemble时）')
    parser.add_argument('--multi_sample', type=int, default=1,
                       help='多次采样取平均（1=不采样，3=采样3次）')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据变换
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
    feature_extractor = MultiModalFeatureExtractor(args.model_name).to(device)
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
    
    if args.classifier == 'ensemble':
        # 简单集成学习：训练多个分类器，平均预测
        print(f"使用集成学习，包含: {args.ensemble_models}")
        classifiers = []
        
        for clf_type in args.ensemble_models:
            print(f"  训练 {clf_type}...")
            if args.tune_params:
                # 尝试不同的超参数
                if clf_type == 'logistic':
                    best_score = 0
                    best_clf = None
                    for C in [0.1, 1.0, 10.0, 100.0]:
                        clf = create_classifier(clf_type, C=C)
                        scores = cross_val_score(clf, train_features, train_labels, cv=5)
                        score = scores.mean()
                        print(f"    C={C}: {score:.4f}")
                        if score > best_score:
                            best_score = score
                            best_clf = clf
                    best_clf.fit(train_features, train_labels)
                    classifiers.append(best_clf)
                    print(f"    最佳C: {best_clf.named_steps['clf'].C}, CV分数: {best_score:.4f}")
                elif clf_type == 'svm':
                    best_score = 0
                    best_clf = None
                    for C in [0.1, 1.0, 10.0]:
                        clf = create_classifier(clf_type, C=C)
                        scores = cross_val_score(clf, train_features, train_labels, cv=5)
                        score = scores.mean()
                        print(f"    C={C}: {score:.4f}")
                        if score > best_score:
                            best_score = score
                            best_clf = clf
                    best_clf.fit(train_features, train_labels)
                    classifiers.append(best_clf)
                    print(f"    最佳C: {best_clf.named_steps['clf'].C}, CV分数: {best_score:.4f}")
                elif clf_type == 'knn':
                    best_score = 0
                    best_clf = None
                    for k in [3, 5, 7, 9]:
                        clf = create_classifier(clf_type, n_neighbors=k)
                        scores = cross_val_score(clf, train_features, train_labels, cv=5)
                        score = scores.mean()
                        print(f"    k={k}: {score:.4f}")
                        if score > best_score:
                            best_score = score
                            best_clf = clf
                    best_clf.fit(train_features, train_labels)
                    classifiers.append(best_clf)
                    print(f"    最佳k: {best_clf.named_steps['clf'].n_neighbors}, CV分数: {best_score:.4f}")
            else:
                # 使用默认参数
                clf = create_classifier(clf_type)
                clf.fit(train_features, train_labels)
                classifiers.append(clf)
                
                # 显示交叉验证分数
                scores = cross_val_score(clf, train_features, train_labels, cv=5)
                print(f"    交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        classifier = classifiers
        print("集成分类器训练完成！")
    else:
        # 单个分类器
        if args.tune_params:
            print("自动调优超参数...")
            if args.classifier == 'logistic':
                best_score = 0
                best_clf = None
                for C in [0.1, 1.0, 10.0, 100.0]:
                    clf = create_classifier(args.classifier, C=C)
                    scores = cross_val_score(clf, train_features, train_labels, cv=5)
                    score = scores.mean()
                    print(f"  C={C}: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_clf = clf
                classifier = best_clf
                classifier.fit(train_features, train_labels)
                print(f"最佳C: {classifier.named_steps['clf'].C}, CV分数: {best_score:.4f}")
            elif args.classifier == 'svm':
                best_score = 0
                best_clf = None
                for C in [0.1, 1.0, 10.0]:
                    clf = create_classifier(args.classifier, C=C)
                    scores = cross_val_score(clf, train_features, train_labels, cv=5)
                    score = scores.mean()
                    print(f"  C={C}: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_clf = clf
                classifier = best_clf
                classifier.fit(train_features, train_labels)
                print(f"最佳C: {classifier.named_steps['clf'].C}, CV分数: {best_score:.4f}")
            elif args.classifier == 'knn':
                best_score = 0
                best_clf = None
                for k in [3, 5, 7, 9]:
                    clf = create_classifier(args.classifier, n_neighbors=k)
                    scores = cross_val_score(clf, train_features, train_labels, cv=5)
                    score = scores.mean()
                    print(f"  k={k}: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_clf = clf
                classifier = best_clf
                classifier.fit(train_features, train_labels)
                print(f"最佳k: {classifier.named_steps['clf'].n_neighbors}, CV分数: {best_score:.4f}")
        else:
            # 使用默认参数（与原版相同）
            classifier = create_classifier(args.classifier)
            classifier.fit(train_features, train_labels)
            
            # 显示交叉验证分数
            scores = cross_val_score(classifier, train_features, train_labels, cv=5)
            print(f"交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        print("分类器训练完成！")
    
    # 提取测试集特征（支持多次采样）
    print("\n提取测试集特征...")
    all_test_proba = []
    
    for sample_idx in range(args.multi_sample):
        if args.multi_sample > 1:
            print(f"  采样 {sample_idx + 1}/{args.multi_sample}...")
        
        test_dataset = MultiModalDataset(
            args.data_root_test, args.video_list_test,
            num_segments=args.num_segments,
            transform=transform,
            random_shift=(sample_idx > 0),  # 第一次不随机，后续随机采样
            test_mode=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        test_features, _, test_video_ids = extract_features(
            feature_extractor, test_loader, device, is_test_mode=True
        )
        
        if sample_idx == 0:
            final_test_video_ids = test_video_ids
            print(f"测试特征形状: {test_features.shape}")
        
        if test_video_ids is None:
            raise ValueError("未能提取到测试集的video_ids")
        
        # 预测
        if args.classifier == 'ensemble':
            proba = simple_ensemble_predict(classifier, test_features)
        else:
            proba = classifier.predict_proba(test_features)
        
        all_test_proba.append(proba)
    
    # 平均多次采样的结果
    if args.multi_sample > 1:
        print(f"\n平均 {args.multi_sample} 次采样的预测结果...")
        predictions_proba = np.mean(all_test_proba, axis=0)
    else:
        predictions_proba = all_test_proba[0]
    
    # 获取Top-5预测
    top5_indices = np.argsort(predictions_proba, axis=1)[:, -5:][:, ::-1]
    
    # 按video_id排序
    sorted_indices = np.argsort(final_test_video_ids)
    top5_indices = top5_indices[sorted_indices]
    final_test_video_ids = final_test_video_ids[sorted_indices]
    
    # 生成提交文件
    prediction_strings = []
    for pred in top5_indices:
        pred_str = ' '.join([str(int(cls)) for cls in pred])
        prediction_strings.append(pred_str)
    
    df = pd.DataFrame({
        'video_id': final_test_video_ids,
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
    if args.classifier == 'ensemble':
        joblib.dump(classifier, classifier_path)
    else:
        joblib.dump(classifier, classifier_path)
    print(f"\n分类器已保存到: {classifier_path}")


if __name__ == '__main__':
    main()








