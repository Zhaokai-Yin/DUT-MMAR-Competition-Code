"""
零样本/少样本推理 - 增强版（带微调选项）
支持多种微调方法提升性能
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
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


def create_classifier(classifier_type, use_pca=False, pca_components=None, 
                     use_grid_search=False, cv_folds=5):
    """创建分类器，支持PCA和网格搜索"""
    steps = []
    
    # 标准化
    steps.append(('scaler', StandardScaler()))
    
    # PCA降维（可选）
    if use_pca and pca_components:
        steps.append(('pca', PCA(n_components=pca_components)))
    
    # 分类器
    if classifier_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
        if use_grid_search:
            param_grid = {
                'clf__C': [0.1, 1.0, 10.0, 100.0],
                'clf__solver': ['lbfgs', 'sag']
            }
    elif classifier_type == 'svm':
        clf = SVC(kernel='rbf', probability=True)
        if use_grid_search:
            param_grid = {
                'clf__C': [0.1, 1.0, 10.0],
                'clf__gamma': ['scale', 'auto', 0.001, 0.01]
            }
    elif classifier_type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5)
        if use_grid_search:
            param_grid = {
                'clf__n_neighbors': [3, 5, 7, 9],
                'clf__weights': ['uniform', 'distance']
            }
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        if use_grid_search:
            param_grid = {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [10, 20, None]
            }
    else:
        raise ValueError(f"不支持的分类器: {classifier_type}")
    
    steps.append(('clf', clf))
    pipeline = Pipeline(steps)
    
    if use_grid_search:
        return GridSearchCV(pipeline, param_grid, cv=cv_folds, n_jobs=-1, verbose=1)
    else:
        return pipeline


def ensemble_predict(classifiers, features, weights=None):
    """集成多个分类器的预测结果
    
    Args:
        classifiers: 分类器列表
        features: 特征
        weights: 权重列表，如果为None则使用等权重
    """
    all_proba = []
    for clf in classifiers:
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(features)
        elif hasattr(clf, 'best_estimator_'):
            # GridSearchCV包装的模型
            proba = clf.best_estimator_.predict_proba(features)
        else:
            # 直接是分类器
            proba = clf.predict_proba(features)
        all_proba.append(proba)
    
    # 加权平均概率
    if weights is None:
        weights = np.ones(len(all_proba)) / len(all_proba)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化
    
    ensemble_proba = np.average(all_proba, axis=0, weights=weights)
    return ensemble_proba


def evaluate_classifier(classifier, features, labels, top_k=5):
    """评估分类器性能"""
    if hasattr(classifier, 'predict_proba'):
        proba = classifier.predict_proba(features)
    elif hasattr(classifier, 'best_estimator_'):
        proba = classifier.best_estimator_.predict_proba(features)
    else:
        proba = classifier.predict_proba(features)
    
    # Top-1准确率
    top1_pred = np.argmax(proba, axis=1)
    top1_acc = np.mean(top1_pred == labels)
    
    # Top-K准确率
    topk_pred = np.argsort(proba, axis=1)[:, -top_k:][:, ::-1]
    topk_acc = np.mean([labels[i] in topk_pred[i] for i in range(len(labels))])
    
    return top1_acc, topk_acc


def main():
    parser = argparse.ArgumentParser(description='零样本/少样本推理 - 增强版')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--model_name', type=str, default='r2plus1d_18',
                       choices=['r2plus1d_18', 'r3d_18', 'mc3_18'],
                       help='预训练模型')
    parser.add_argument('--classifier', type=str, default='logistic',
                       choices=['logistic', 'svm', 'knn', 'rf', 'ensemble'],
                       help='分类器类型')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_segments', type=int, default=16,
                       help='视频分段数量')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--gpu', type=int, default=0)
    
    # 微调选项
    parser.add_argument('--use_pca', action='store_true',
                       help='使用PCA降维')
    parser.add_argument('--pca_components', type=int, default=512,
                       help='PCA降维后的维度')
    parser.add_argument('--use_grid_search', action='store_true',
                       help='使用网格搜索优化超参数')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='交叉验证折数')
    parser.add_argument('--ensemble_models', type=str, nargs='+',
                       default=['logistic', 'svm', 'rf'],
                       help='集成学习的分类器列表')
    parser.add_argument('--multi_model_ensemble', action='store_true',
                       help='使用多个预训练模型集成')
    parser.add_argument('--eval_on_val', action='store_true',
                       help='在验证集上评估性能')
    
    args = parser.parse_args()
    
    # 全局变量用于存储集成权重
    classifier_weights = None
    
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
    
    # 特征工程
    pca = None
    if args.use_pca:
        # 确保PCA组件数不超过样本数和特征数
        max_components = min(train_features.shape[0], train_features.shape[1], args.pca_components)
        if max_components < args.pca_components:
            print(f"\n警告: PCA组件数从 {args.pca_components} 调整为 {max_components} (受限于样本数 {train_features.shape[0]})")
        
        print(f"\n使用PCA降维到 {max_components} 维...")
        pca = PCA(n_components=max_components)
        train_features = pca.fit_transform(train_features)
        print(f"降维后特征形状: {train_features.shape}")
        print(f"解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    
    # 训练分类器
    print(f"\n训练{args.classifier}分类器...")
    
    if args.classifier == 'ensemble':
        # 集成学习
        print(f"使用集成学习，包含: {args.ensemble_models}")
        classifiers = []
        classifier_weights = []
        # 如果已经在主流程中做了PCA，分类器中就不应该再做了
        use_pca_in_classifier = False if args.use_pca else args.use_pca
        
        # 划分验证集用于评估权重（但不在上面训练，只在完整数据上训练）
        from sklearn.model_selection import train_test_split
        train_features_split, val_features_split, train_labels_split, val_labels_split = train_test_split(
            train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # 先在验证集上训练临时模型来评估权重
        temp_classifiers = []
        for clf_type in args.ensemble_models:
            print(f"  评估 {clf_type}...")
            temp_clf = create_classifier(clf_type, use_pca_in_classifier, args.pca_components, 
                                       False, args.cv_folds)  # 不使用网格搜索，快速评估
            temp_clf.fit(train_features_split, train_labels_split)
            temp_classifiers.append(temp_clf)
            
            # 在验证集上评估
            top1_acc, top5_acc = evaluate_classifier(temp_clf, val_features_split, val_labels_split)
            print(f"    {clf_type} 验证集 Top-1准确率: {top1_acc:.4f}, Top-5准确率: {top5_acc:.4f}")
            
            # 使用Top-1准确率作为权重
            classifier_weights.append(top1_acc)
        
        # 在完整训练集上训练最终模型
        print("\n在完整训练集上训练最终分类器...")
        for clf_type in args.ensemble_models:
            print(f"  训练 {clf_type}...")
            clf = create_classifier(clf_type, use_pca_in_classifier, args.pca_components, 
                                  args.use_grid_search, args.cv_folds)
            clf.fit(train_features, train_labels)  # 使用完整训练集
            classifiers.append(clf)
        
        classifier = classifiers
        print(f"集成分类器训练完成！权重: {dict(zip(args.ensemble_models, classifier_weights))}")
    else:
        classifier_weights = None
        # 如果已经在主流程中做了PCA，分类器中就不应该再做了
        use_pca_in_classifier = False if args.use_pca else args.use_pca
        classifier = create_classifier(args.classifier, use_pca_in_classifier, args.pca_components,
                                     args.use_grid_search, args.cv_folds)
        
        # 如果启用验证集评估，先划分数据
        if args.eval_on_val:
            from sklearn.model_selection import train_test_split
            train_features_split, val_features_split, train_labels_split, val_labels_split = train_test_split(
                train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
            )
            classifier.fit(train_features_split, train_labels_split)
            
            # 在验证集上评估
            top1_acc, top5_acc = evaluate_classifier(classifier, val_features_split, val_labels_split)
            print(f"验证集 Top-1准确率: {top1_acc:.4f}, Top-5准确率: {top5_acc:.4f}")
            
            # 在完整训练集上重新训练
            print("在完整训练集上重新训练...")
            classifier.fit(train_features, train_labels)
        else:
            classifier.fit(train_features, train_labels)
        
        if args.use_grid_search:
            print(f"最佳参数: {classifier.best_params_}")
            print(f"最佳交叉验证分数: {classifier.best_score_:.4f}")
        else:
            scores = cross_val_score(classifier, train_features, train_labels, cv=args.cv_folds)
            print(f"交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
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
    
    if test_video_ids is None:
        raise ValueError("未能提取到测试集的video_ids")
    
    # 对测试特征应用PCA（如果使用了）
    if args.use_pca and pca is not None:
        test_features = pca.transform(test_features)
    
    # 预测
    print("\n进行预测...")
    if args.classifier == 'ensemble':
        # 使用加权集成
        predictions_proba = ensemble_predict(classifier, test_features, weights=classifier_weights)
    else:
        if args.use_grid_search:
            predictions_proba = classifier.best_estimator_.predict_proba(test_features)
        else:
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
    
    # 保存分类器和PCA（如果使用了）
    classifier_path = args.output.replace('.csv', '_classifier.pkl')
    if args.classifier == 'ensemble':
        joblib.dump(classifier, classifier_path)
    else:
        joblib.dump(classifier, classifier_path)
    
    if args.use_pca and pca is not None:
        pca_path = args.output.replace('.csv', '_pca.pkl')
        joblib.dump(pca, pca_path)
        print(f"PCA模型已保存到: {pca_path}")
    
    print(f"\n分类器已保存到: {classifier_path}")


if __name__ == '__main__':
    main()

