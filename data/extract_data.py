"""
解压MMAR数据集
"""
import os
import zipfile
from tqdm import tqdm

def extract_zip(zip_path, extract_to):
    """解压zip文件"""
    if not os.path.exists(zip_path):
        print(f"文件不存在: {zip_path}")
        return False
    
    print(f"正在解压: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 获取文件列表
        file_list = zip_ref.namelist()
        # 解压所有文件
        for file in tqdm(file_list, desc=f"解压 {os.path.basename(zip_path)}"):
            zip_ref.extract(file, extract_to)
    print(f"解压完成: {zip_path}")
    return True

def main():
    base_dir = "MMAR"
    
    # 训练集
    train_dir = os.path.join(base_dir, "train_500")
    
    print("=" * 50)
    print("解压训练集数据")
    print("=" * 50)
    
    # RGB数据
    rgb_zip = os.path.join(train_dir, "rgb_data.zip")
    rgb_extract = os.path.join(train_dir, "rgb_data")
    if not os.path.exists(rgb_extract):
        os.makedirs(rgb_extract, exist_ok=True)
    if os.path.exists(rgb_zip):
        # zip文件内结构是：500/000001.jpg，需要解压到rgb_data目录
        extract_zip(rgb_zip, rgb_extract)
    
    # IR数据
    ir_zip = os.path.join(train_dir, "ir_data.zip")
    ir_extract = os.path.join(train_dir, "ir_data")
    if not os.path.exists(ir_extract):
        os.makedirs(ir_extract, exist_ok=True)
    if os.path.exists(ir_zip):
        # zip文件内结构是：500/000001.jpg，需要解压到ir_data目录
        extract_zip(ir_zip, ir_extract)
    
    # Depth数据（多个zip文件）
    depth_dir = os.path.join(train_dir, "depth_data")
    if os.path.exists(depth_dir):
        depth_zips = [f for f in os.listdir(depth_dir) if f.endswith('.zip')]
        depth_extract = os.path.join(train_dir, "depth_data")
        if not os.path.exists(depth_extract):
            os.makedirs(depth_extract, exist_ok=True)
        
        for depth_zip in sorted(depth_zips):
            depth_zip_path = os.path.join(depth_dir, depth_zip)
            extract_zip(depth_zip_path, depth_dir)
    
    # 测试集
    test_dir = os.path.join(base_dir, "test_200")
    
    print("\n" + "=" * 50)
    print("解压测试集数据")
    print("=" * 50)
    
    # RGB数据
    rgb_zip = os.path.join(test_dir, "rgb_data.zip")
    rgb_extract = os.path.join(test_dir, "rgb_data")
    if not os.path.exists(rgb_extract):
        os.makedirs(rgb_extract, exist_ok=True)
    if os.path.exists(rgb_zip):
        extract_zip(rgb_zip, rgb_extract)
    
    # IR数据
    ir_zip = os.path.join(test_dir, "ir_data.zip")
    ir_extract = os.path.join(test_dir, "ir_data")
    if not os.path.exists(ir_extract):
        os.makedirs(ir_extract, exist_ok=True)
    if os.path.exists(ir_zip):
        extract_zip(ir_zip, ir_extract)
    
    # Depth数据（多个zip文件）
    depth_dir = os.path.join(test_dir, "depth_data")
    if os.path.exists(depth_dir):
        depth_zips = [f for f in os.listdir(depth_dir) if f.endswith('.zip')]
        depth_extract = os.path.join(test_dir, "depth_data")
        if not os.path.exists(depth_extract):
            os.makedirs(depth_extract, exist_ok=True)
        
        for depth_zip in sorted(depth_zips):
            depth_zip_path = os.path.join(depth_dir, depth_zip)
            extract_zip(depth_zip_path, depth_dir)
    
    print("\n" + "=" * 50)
    print("数据解压完成！")
    print("=" * 50)
    
    # 检查解压后的结构
    print("\n检查数据目录结构...")
    for dataset_type in ["train_500", "test_200"]:
        dataset_dir = os.path.join(base_dir, dataset_type)
        for modality in ["rgb_data", "ir_data", "depth_data"]:
            modality_dir = os.path.join(dataset_dir, modality)
            if os.path.exists(modality_dir):
                # 检查是否有video目录
                video_dirs = [d for d in os.listdir(modality_dir) if d.startswith('video_')]
                print(f"{dataset_type}/{modality}: {len(video_dirs)} 个视频目录")

if __name__ == '__main__':
    main()

