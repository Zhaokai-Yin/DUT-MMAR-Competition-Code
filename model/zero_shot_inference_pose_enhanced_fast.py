"""
快速版本：减少backbone数量，关闭骨骼点检测
用于快速测试和调试
"""
import os
import sys

# 复制主脚本的参数，但使用更快的配置
if __name__ == '__main__':
    # 快速配置：只使用ConvNeXt Large，关闭骨骼点
    cmd = [
        'python', 'zero_shot_inference_pose_enhanced.py',
        '--backbone_names', 'convnext_large',
        '--classifier', 'ensemble',
        '--ensemble_models', 'logistic,svm,rf',
        '--tta_times', '3',
        '--output', 'submission_pose_fast_test.csv',
        '--gpu', '0'
    ]
    
    print("=" * 50)
    print("快速测试配置")
    print("=" * 50)
    print("Backbone: ConvNeXt Large only")
    print("骨骼点检测: 关闭")
    print("TTA: 3次")
    print("=" * 50)
    print()
    
    import subprocess
    subprocess.run(cmd)




