"""
VideoMAE V2版本 - 使用最新的视频理解模型
VideoMAE V2是专门为视频设计的模型，比图像模型更适合视频行为识别
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

# 尝试导入transformers库（用于VideoMAE V2）
try:
    from transformers import AutoModel, AutoConfig, AutoImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("警告: transformers库未安装，请运行: pip install transformers")
    print("将使用torchvision的视频模型作为备选")

# 尝试导入easydict（VideoMAE V2需要）
try:
    import easydict
    HAS_EASYDICT = True
except ImportError:
    HAS_EASYDICT = False
    print("警告: easydict库未安装，VideoMAE V2需要此库")
    print("请运行: pip install easydict")


class VideoMAEV2Extractor(nn.Module):
    """VideoMAE V2特征提取器"""
    def __init__(self, model_name='videomae_v2_base', num_frames=16):
        super(VideoMAEV2Extractor, self).__init__()
        self.model_name = model_name
        self.num_frames = num_frames
        
        if not HAS_TRANSFORMERS:
            raise ImportError("需要安装transformers库: pip install transformers")
        
        if not HAS_EASYDICT:
            print("警告: easydict未安装，VideoMAE V2可能无法正常工作")
            print("请运行: pip install easydict")
        
        # VideoMAE V2模型配置
        model_configs = {
            'videomae_v2_base': 'OpenGVLab/VideoMAEv2-Base',
            'videomae_v2_large': 'OpenGVLab/VideoMAEv2-Large',
            'videomae_v2_giant': 'OpenGVLab/VideoMAEv2-Giant',
        }
        
        # 检查是否是VideoMAE V2模型
        is_videomae = model_name in model_configs
        
        if is_videomae:
            # 使用VideoMAE V2
            hf_model_name = model_configs[model_name]
            try:
                print(f"加载VideoMAE V2模型: {hf_model_name}")
                # 加载配置
                config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
                # 加载模型
                self.model = AutoModel.from_pretrained(hf_model_name, config=config, trust_remote_code=True)
                # 加载图像处理器
                self.processor = AutoImageProcessor.from_pretrained(hf_model_name, trust_remote_code=True)
                
                # 获取特征维度
                if hasattr(config, 'hidden_size'):
                    self.feature_dim = config.hidden_size
                elif hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                    self.feature_dim = self.model.config.hidden_size
                else:
                    # 默认值
                    if 'base' in model_name:
                        self.feature_dim = 768
                    elif 'large' in model_name:
                        self.feature_dim = 1024
                    elif 'giant' in model_name:
                        self.feature_dim = 1408
                    else:
                        self.feature_dim = 768
                
                self.model.eval()
                print(f"VideoMAE V2模型加载成功，特征维度: {self.feature_dim}")
            except Exception as e:
                print(f"加载VideoMAE V2失败: {e}")
                if "easydict" in str(e).lower():
                    print("错误: 缺少easydict库")
                    print("请运行: pip install easydict")
                raise e
        else:
            # 使用torchvision视频模型
            import torchvision.models.video as video_models
            print(f"使用torchvision视频模型: {model_name}")
            
            if model_name == 'r2plus1d_18':
                self.model = video_models.r2plus1d_18(pretrained=True)
                self.feature_dim = 512
            elif model_name == 'r3d_18':
                self.model = video_models.r3d_18(pretrained=True)
                self.feature_dim = 512
            elif model_name == 'mc3_18':
                self.model = video_models.mc3_18(pretrained=True)
                self.feature_dim = 512
            else:
                # 默认使用R2Plus1D-18
                self.model = video_models.r2plus1d_18(pretrained=True)
                self.feature_dim = 512
            
            self.model.fc = nn.Identity()  # 移除分类层
            self.processor = None  # 不使用processor
            self.model.eval()
            print(f"模型加载成功，特征维度: {self.feature_dim}")
    
    def forward(self, x):
        """
        x: [B, C, T, H, W] - 视频帧序列，已经归一化到[0,1]
        """
        B, C, T, H, W = x.size()
        
        if self.processor is not None:
            # 使用VideoMAE V2
            all_features = []
            with torch.no_grad():
                for b in range(B):
                    # 处理每个batch的视频
                    video_frames = []
                    for t in range(T):
                        # 转换为PIL Image格式
                        frame_tensor = x[b, :, t, :, :]  # [C, H, W]
                        # 反归一化到[0, 1]
                        frame_tensor = torch.clamp(frame_tensor, 0, 1)
                        # 转换为numpy并转置为[H, W, C]
                        frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
                        # 转换为0-255范围
                        frame_np = (frame_np * 255).astype(np.uint8)
                        # 转换为PIL Image
                        from PIL import Image
                        if C == 3:
                            frame_pil = Image.fromarray(frame_np, mode='RGB')
                        else:
                            frame_pil = Image.fromarray(frame_np.squeeze(), mode='L').convert('RGB')
                        video_frames.append(frame_pil)
                    
                    # 使用processor处理
                    try:
                        inputs = self.processor(video_frames, return_tensors="pt")
                        # 移动到正确的设备
                        inputs = {k: v.to(x.device) for k, v in inputs.items()}
                        
                        # 检查pixel_values的格式并转换
                        if 'pixel_values' in inputs:
                            pixel_values = inputs['pixel_values']
                            # VideoMAE V2期望格式: [B, C, T, H, W]
                            # processor可能返回: [B, T, C, H, W] 或其他格式
                            if len(pixel_values.shape) == 5:
                                # 检查维度顺序
                                if pixel_values.shape[1] == 3:  # 已经是 [B, C, T, H, W]
                                    pass  # 格式正确
                                elif pixel_values.shape[2] == 3:  # 是 [B, T, C, H, W]
                                    # 转换为 [B, C, T, H, W]
                                    pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
                                inputs['pixel_values'] = pixel_values
                        
                        # 提取特征
                        outputs = self.model(**inputs)
                        
                        # 获取特征
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            feat = outputs.pooler_output
                        elif hasattr(outputs, 'last_hidden_state'):
                            # 对时间维度求平均（如果存在）
                            if len(outputs.last_hidden_state.shape) > 2:
                                feat = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]
                            else:
                                feat = outputs.last_hidden_state
                        elif isinstance(outputs, tuple) and len(outputs) > 0:
                            feat = outputs[0]
                            if len(feat.shape) > 2:
                                feat = feat.mean(dim=1)
                        else:
                            # 尝试从字典中获取
                            if isinstance(outputs, dict):
                                if 'pooler_output' in outputs:
                                    feat = outputs['pooler_output']
                                elif 'last_hidden_state' in outputs:
                                    feat = outputs['last_hidden_state']
                                    if len(feat.shape) > 2:
                                        feat = feat.mean(dim=1)
                                else:
                                    feat = list(outputs.values())[0]
                                    if len(feat.shape) > 2:
                                        feat = feat.mean(dim=1)
                            else:
                                # 最后尝试：直接使用第一个输出
                                feat = outputs[0] if isinstance(outputs, tuple) else outputs
                                if len(feat.shape) > 2:
                                    feat = feat.mean(dim=1)
                        
                        # 确保特征维度正确
                        if len(feat.shape) > 2:
                            feat = feat.view(feat.shape[0], -1)
                        
                        all_features.append(feat)
                    except Exception as e:
                        print(f"处理视频时出错 (batch {b}): {e}")
                        import traceback
                        traceback.print_exc()
                        # 使用零特征作为备选
                        all_features.append(torch.zeros(1, self.feature_dim, device=x.device))
                
                if len(all_features) > 0:
                    features = torch.cat(all_features, dim=0)
                else:
                    features = torch.zeros(B, self.feature_dim, device=x.device)
                return features
        else:
            # 使用torchvision视频模型
            # 确保帧数是16的倍数
            if T < 16:
                # 重复帧
                repeat_times = (16 + T - 1) // T
                x = x.repeat(1, 1, repeat_times, 1, 1)[:, :, :16, :, :]
            elif T > 16:
                # 均匀采样16帧
                indices = torch.linspace(0, T - 1, 16).long()
                x = x[:, :, indices, :, :]
            
            with torch.no_grad():
                # R2Plus1D期望输入: [B, C, T, H, W]
                features = self.model(x)
            
            return features


class MultiModalVideoMAEExtractor(nn.Module):
    """多模态VideoMAE V2特征提取器"""
    def __init__(self, model_name='videomae_v2_base', num_frames=16):
        super(MultiModalVideoMAEExtractor, self).__init__()
        # 只创建一个提取器，三个模态共享（更高效）
        # 如果VideoMAE V2加载失败，会使用R2Plus1D作为备选
        print(f"\n为RGB模态加载模型...")
        self.rgb_extractor = VideoMAEV2Extractor(model_name, num_frames)
        print(f"为Depth模态加载模型...")
        self.depth_extractor = VideoMAEV2Extractor(model_name, num_frames)
        print(f"为IR模态加载模型...")
        self.ir_extractor = VideoMAEV2Extractor(model_name, num_frames)
        self.feature_dim = self.rgb_extractor.feature_dim * 3
        print(f"所有模型加载完成，总特征维度: {self.feature_dim}")
    
    def forward(self, rgb_input, depth_input, ir_input):
        rgb_feat = self.rgb_extractor(rgb_input)
        depth_feat = self.depth_extractor(depth_input)
        ir_feat = self.ir_extractor(ir_input)
        # 拼接特征
        combined_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return combined_feat


def extract_features(model, dataloader, device, is_test_mode=False, tta_times=1):
    """提取所有样本的特征，支持TTA"""
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
                features = model(rgb_input, depth_input, ir_input)
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
    parser = argparse.ArgumentParser(description='VideoMAE V2版本 - 使用最新的视频理解模型')
    parser.add_argument('--data_root_train', type=str, default='MMAR/train_500')
    parser.add_argument('--video_list_train', type=str, default='MMAR/train_500/train_videofolder_500.txt')
    parser.add_argument('--data_root_test', type=str, default='MMAR/test_200')
    parser.add_argument('--video_list_test', type=str, default='MMAR/test_200/test_videofolder_200.txt')
    parser.add_argument('--model_name', type=str, default='videomae_v2_base',
                       choices=['videomae_v2_base', 'videomae_v2_large', 'videomae_v2_giant', 'r2plus1d_18', 'r3d_18', 'mc3_18'],
                       help='视频模型版本（VideoMAE V2或torchvision模型）')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='视频帧数（VideoMAE V2通常使用16帧）')
    parser.add_argument('--classifier', type=str, default='svm',
                       choices=['logistic', 'svm', 'rf', 'gbdt', 'ensemble'],
                       help='分类器类型')
    parser.add_argument('--ensemble_models', type=str, default='logistic,svm,rf',
                       help='集成学习的分类器列表（用逗号分隔）')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_segments', type=int, default=16,
                       help='视频分段数量（建议与num_frames相同）')
    parser.add_argument('--output', type=str, default='submission_videomae.csv')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tta_times', type=int, default=1,
                       help='测试时增强次数')
    parser.add_argument('--use_pca', action='store_true',
                       help='使用PCA降维')
    parser.add_argument('--pca_components', type=int, default=1000,
                       help='PCA降维后的维度')
    parser.add_argument('--tune_params', action='store_true',
                       help='自动调优超参数')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='交叉验证折数')
    args = parser.parse_args()
    
    if not HAS_TRANSFORMERS:
        print("错误: 需要安装transformers库")
        print("请运行: pip install transformers")
        return
    
    if not HAS_EASYDICT:
        print("警告: easydict库未安装，VideoMAE V2需要此库")
        print("请运行: pip install easydict")
        print("或者安装所有依赖: pip install -r requirements.txt")
        print("\n继续运行将使用R2Plus1D-18作为备选模型...")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            return
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB')
    
    # 数据变换（VideoMAE V2使用224x224）
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建特征提取模型
    print(f"\n加载视频模型: {args.model_name}")
    try:
        feature_extractor = MultiModalVideoMAEExtractor(
            model_name=args.model_name,
            num_frames=args.num_frames
        ).to(device)
        feature_extractor.eval()
        print(f"总特征维度: {feature_extractor.feature_dim}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保已安装torchvision: pip install torchvision")
        return
    
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
        tta_times=1
    )
    print(f"训练特征形状: {train_features.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    
    # 特征预处理
    preprocessing_steps = [('scaler', StandardScaler())]
    
    if args.use_pca:
        print(f"\n应用PCA降维: {train_features.shape[1]} -> {args.pca_components}")
        preprocessing_steps.append(('pca', PCA(n_components=args.pca_components, random_state=42)))
    
    # 训练分类器
    print(f"\n训练{args.classifier}分类器...")
    
    if args.classifier == 'ensemble':
        # 集成学习
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
        
        voting_clf = VotingClassifier(estimators=classifiers, voting='soft')
        classifier = Pipeline(preprocessing_steps + [('clf', voting_clf)])
        
        if args.tune_params:
            print("使用交叉验证评估集成分类器...")
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(classifier, train_features, train_labels, cv=cv, scoring='accuracy', n_jobs=-1)
            print(f"交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    elif args.classifier == 'logistic':
        classifier = Pipeline(preprocessing_steps + [
            ('clf', LogisticRegression(max_iter=5000, multi_class='multinomial', 
                                     solver='lbfgs', C=2.0, n_jobs=-1))
        ])
    elif args.classifier == 'svm':
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

