#!/bin/bash

# MMAR多模态行为识别快速开始脚本

echo "=== MMAR多模态行为识别 ==="
echo ""

# 检查数据是否存在
if [ ! -d "MMAR/train_500" ]; then
    echo "错误: 未找到训练数据目录 MMAR/train_500"
    echo "请先下载并解压数据集到 MMAR/ 目录"
    exit 1
fi

if [ ! -d "MMAR/test_200" ]; then
    echo "错误: 未找到测试数据目录 MMAR/test_200"
    echo "请先下载并解压数据集到 MMAR/ 目录"
    exit 1
fi

# 创建检查点目录
mkdir -p checkpoints

echo "开始训练..."
python train.py \
    --data_root MMAR/train_500 \
    --video_list MMAR/train_500/train_videofolder_500.txt \
    --num_segments 8 \
    --batch_size 4 \
    --epochs 50 \
    --lr 0.001 \
    --base_model resnet18 \
    --fusion_method late \
    --val_ratio 0.2 \
    --save_dir checkpoints \
    --gpu 0

echo ""
echo "训练完成！"
echo ""

echo "生成提交文件..."
python generate_submission.py \
    --data_root MMAR/test_200 \
    --video_list MMAR/test_200/test_videofolder_200.txt \
    --checkpoint checkpoints/best_model.pth \
    --output submission.csv \
    --gpu 0

echo ""
echo "完成！提交文件已保存为 submission.csv"




