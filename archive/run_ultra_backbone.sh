#!/bin/bash
# 使用超强backbone进行零样本推理（Linux/Mac脚本）
# 推荐配置：ConvNeXt-Base（平衡性能和速度）

echo "========================================"
echo "超强Backbone零样本推理"
echo "========================================"
echo ""

# 设置参数
BACKBONE="convnext"
MODEL_SIZE="base"
BATCH_SIZE=4
OUTPUT="submission_ultra_convnext_base.csv"

echo "使用Backbone: $BACKBONE ($MODEL_SIZE)"
echo "批次大小: $BATCH_SIZE"
echo "输出文件: $OUTPUT"
echo ""

# 运行推理
python zero_shot_inference_ultra.py \
    --backbone $BACKBONE \
    --model_size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --output $OUTPUT \
    --gpu 0

echo ""
echo "========================================"
echo "推理完成！"
echo "结果保存在: $OUTPUT"
echo "========================================"






