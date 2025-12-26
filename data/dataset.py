import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


class MMARDataset(Dataset):
    """MMAR多模态数据集加载器"""
    
    def __init__(self, data_root, video_list_file, num_segments=8, 
                 new_length=1, modality='RGB', transform=None, 
                 random_shift=True, test_mode=False):
        """
        Args:
            data_root: 数据根目录
            video_list_file: 视频列表文件路径
            num_segments: 视频分段数量
            new_length: 每个分段采样帧数
            modality: 模态类型 ('RGB', 'Depth', 'IR')
            transform: 数据增强变换
            random_shift: 是否随机采样
            test_mode: 是否为测试模式
        """
        self.data_root = data_root
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        
        # 读取视频列表
        self.video_list = []
        with open(video_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                video_id = int(parts[0])
                frame_count = int(parts[1])
                
                if not test_mode and len(parts) >= 3:
                    label = int(parts[2])
                else:
                    label = -1  # 测试集没有标签
                
                # 检测该模态的实际帧数
                actual_frame_count = self._detect_frame_count(video_id, frame_count)
                
                self.video_list.append({
                    'video_id': video_id,
                    'frame_count': actual_frame_count,  # 使用实际检测到的帧数
                    'label': label
                })
    
    def _detect_frame_count(self, video_id, default_count):
        """检测指定视频在该模态下的实际帧数"""
        # 确定数据目录
        if self.modality == 'RGB':
            possible_dirs = [
                os.path.join(self.data_root, str(video_id)),
                os.path.join(self.data_root, 'rgb_data', str(video_id)),
                os.path.join(self.data_root, 'rgb_data', f'video_{video_id}'),
            ]
        elif self.modality == 'Depth':
            possible_dirs = [
                os.path.join(self.data_root, 'depth_data', str(video_id)),
                os.path.join(self.data_root, 'depth_data', f'video_{video_id}'),
            ]
        elif self.modality == 'IR':
            possible_dirs = [
                os.path.join(self.data_root, 'ir_data', str(video_id)),
                os.path.join(self.data_root, 'ir_data', f'video_{video_id}'),
            ]
        else:
            return default_count
        
        # 找到存在的目录
        video_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                video_dir = dir_path
                break
        
        if video_dir is None:
            # 目录不存在，返回默认值
            return default_count
        
        # 统计目录中的图像文件数量
        try:
            files = os.listdir(video_dir)
            # 统计图像文件（支持多种格式）
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                # 尝试从文件名中提取最大索引
                max_index = 0
                for f in image_files:
                    # 尝试提取数字（支持000001.jpg或1.jpg格式）
                    base_name = os.path.splitext(f)[0]
                    # 移除可能的frame_前缀
                    if base_name.startswith('frame_'):
                        base_name = base_name[6:]
                    try:
                        idx = int(base_name)
                        max_index = max(max_index, idx)
                    except:
                        pass
                
                if max_index > 0:
                    return max_index
                else:
                    return len(image_files)
            else:
                return default_count
        except Exception as e:
            # 如果检测失败，返回默认值
            return default_count
    
    def _load_image(self, directory, index):
        """加载单张图像"""
        # 确保索引有效
        if index < 1:
            raise ValueError(f"无效的帧索引: {index} (必须 >= 1)")
        
        # 尝试不同的文件名格式（优先尝试6位数字，因为实际文件是000001.jpg格式）
        possible_paths = [
            os.path.join(directory, f'{index:06d}.jpg'),  # 6位数字，如000005.jpg
            os.path.join(directory, f'{index:06d}.png'),
            os.path.join(directory, f'{index:06d}.jpeg'),
            os.path.join(directory, f'{index:05d}.jpg'),  # 5位数字，如00005.jpg
            os.path.join(directory, f'{index:05d}.png'),
            os.path.join(directory, f'{index:05d}.jpeg'),
            os.path.join(directory, f'{index}.jpg'),  # 无前导零，如5.jpg
            os.path.join(directory, f'{index}.png'),
            os.path.join(directory, f'frame_{index:06d}.jpg'),
            os.path.join(directory, f'frame_{index:05d}.jpg'),
            os.path.join(directory, f'frame_{index}.jpg'),
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            # 提供更详细的错误信息
            abs_dir = os.path.abspath(directory)
            if not os.path.exists(abs_dir):
                raise ValueError(f"目录不存在: {abs_dir}")
            
            # 列出目录中的一些文件作为参考
            try:
                files = sorted(os.listdir(abs_dir))[:10]
                file_list = ', '.join(files)
            except:
                file_list = "无法列出文件"
            
            raise ValueError(
                f"无法找到图像文件\n"
                f"  目录: {abs_dir}\n"
                f"  索引: {index}\n"
                f"  尝试的路径: {possible_paths[:3]}...\n"
                f"  目录中的文件示例: {file_list}"
            )
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        
        if self.modality == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.modality == 'Depth':
            # Depth图像可能是灰度图或彩色图
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.modality == 'IR':
            # IR图像可能是灰度图或彩色图
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _sample_indices(self, num_frames):
        """采样帧索引"""
        if num_frames <= 0:
            raise ValueError(f"无效的帧数: {num_frames}")
        
        if self.test_mode:
            # 测试模式：均匀采样
            if num_frames > self.num_segments * self.new_length:
                tick = (num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.sort(np.random.choice(num_frames, self.num_segments))
        else:
            # 训练模式：随机采样
            if self.random_shift:
                if num_frames > self.num_segments * self.new_length:
                    tick = (num_frames - self.new_length + 1) / float(self.num_segments)
                    offsets = np.array([int(tick / 2.0 + tick * x) + random.randint(-int(tick/2), int(tick/2)) 
                                       for x in range(self.num_segments)])
                    offsets = np.clip(offsets, 0, num_frames - self.new_length)
                else:
                    offsets = np.sort(np.random.choice(num_frames, self.num_segments))
            else:
                # 均匀采样
                if num_frames > self.num_segments * self.new_length:
                    tick = (num_frames - self.new_length + 1) / float(self.num_segments)
                    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
                else:
                    offsets = np.sort(np.random.choice(num_frames, self.num_segments))
        
        # 确保所有索引都在有效范围内
        offsets = np.clip(offsets, 0, max(0, num_frames - 1))
        return offsets
    
    def __getitem__(self, index):
        video_info = self.video_list[index]
        video_id = video_info['video_id']
        num_frames = video_info['frame_count']
        label = video_info['label']
        
        # 确定数据目录（支持多种格式，按优先级排序）
        if self.modality == 'RGB':
            # 尝试多种可能的路径（优先检查根目录下的数字目录）
            possible_dirs = [
                os.path.join(self.data_root, str(video_id)),  # RGB数据可能在根目录下（最常见）
                os.path.join(self.data_root, 'rgb_data', str(video_id)),
                os.path.join(self.data_root, 'rgb_data', f'video_{video_id}'),
            ]
        elif self.modality == 'Depth':
            possible_dirs = [
                os.path.join(self.data_root, 'depth_data', str(video_id)),
                os.path.join(self.data_root, 'depth_data', f'video_{video_id}'),
            ]
        elif self.modality == 'IR':
            possible_dirs = [
                os.path.join(self.data_root, 'ir_data', str(video_id)),
                os.path.join(self.data_root, 'ir_data', f'video_{video_id}'),
            ]
        else:
            raise ValueError(f"未知模态: {self.modality}")
        
        # 找到存在的目录
        video_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                video_dir = dir_path
                break
        
        if video_dir is None:
            raise ValueError(f"视频目录不存在 (video_id={video_id}, modality={self.modality}): 尝试了 {possible_dirs}")
        
        # 采样帧索引
        frame_indices = self._sample_indices(num_frames)
        
        # 确保所有索引都在有效范围内
        frame_indices = np.clip(frame_indices, 0, num_frames - 1)
        
        # 加载图像
        images = []
        for idx in frame_indices:
            for i in range(self.new_length):
                frame_idx = min(int(idx) + i, num_frames - 1)
                # 确保frame_idx在有效范围内，并且转换为1-based索引
                frame_idx = max(0, min(frame_idx, num_frames - 1))
                img = self._load_image(video_dir, frame_idx + 1)  # 帧编号从1开始
                images.append(img)
        
        # 转换为tensor
        if self.transform:
            images = [self.transform(img) for img in images]
        else:
            # 默认转换
            transform_default = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            images = [transform_default(img) for img in images]
        
        # 堆叠为 [C, T, H, W]
        images = torch.stack(images, dim=1)
        
        if self.test_mode:
            return images, video_id
        else:
            return images, label
    
    def __len__(self):
        return len(self.video_list)


class MultiModalDataset(Dataset):
    """多模态数据集，同时加载RGB、Depth、IR"""
    
    def __init__(self, data_root, video_list_file, num_segments=8, 
                 new_length=1, transform=None, random_shift=True, test_mode=False):
        self.rgb_dataset = MMARDataset(data_root, video_list_file, num_segments, 
                                      new_length, 'RGB', transform, random_shift, test_mode)
        self.depth_dataset = MMARDataset(data_root, video_list_file, num_segments, 
                                        new_length, 'Depth', transform, random_shift, test_mode)
        self.ir_dataset = MMARDataset(data_root, video_list_file, num_segments, 
                                     new_length, 'IR', transform, random_shift, test_mode)
        self.test_mode = test_mode
    
    def __getitem__(self, index):
        rgb_data = self.rgb_dataset[index]
        depth_data = self.depth_dataset[index]
        ir_data = self.ir_dataset[index]
        
        if self.test_mode:
            rgb_img, video_id = rgb_data
            depth_img, _ = depth_data
            ir_img, _ = ir_data
            return (rgb_img, depth_img, ir_img), video_id
        else:
            rgb_img, label = rgb_data
            depth_img, _ = depth_data
            ir_img, _ = ir_data
            return (rgb_img, depth_img, ir_img), label
    
    def __len__(self):
        return len(self.rgb_dataset)

