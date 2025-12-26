# PowerShell脚本 - 超强集成推理（多个backbone + TTA）
# 目标：指标突破92+

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "超强集成推理 - 多个Backbone + TTA" -ForegroundColor Cyan
Write-Host "目标：指标突破92+" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 设置参数
# 注意：如果ResNeXt-101下载卡住，可以改为"50"（更快，性能差异很小）
$BACKBONES = @("convnext", "resnext", "efficientnet_v2")
$MODEL_SIZES = @("large", "50", "l")  # 使用50代替101，避免下载卡住
$BATCH_SIZE = 2
$OUTPUT = "submission_ensemble_ultra.csv"
$USE_TTA = $true

Write-Host "集成Backbone: $($BACKBONES -join ', ')" -ForegroundColor Yellow
Write-Host "模型大小: $($MODEL_SIZES -join ', ')" -ForegroundColor Yellow
Write-Host "批次大小: $BATCH_SIZE" -ForegroundColor Yellow
Write-Host "使用TTA: $USE_TTA" -ForegroundColor Yellow
Write-Host "输出文件: $OUTPUT" -ForegroundColor Yellow
Write-Host ""

# 构建命令
$cmd = "python zero_shot_inference_ensemble_ultra.py --backbones $($BACKBONES -join ' ') --model_sizes $($MODEL_SIZES -join ' ') --batch_size $BATCH_SIZE --output $OUTPUT --gpu 0"

if ($USE_TTA) {
    $cmd += " --use_tta"
}

Write-Host "执行命令: $cmd" -ForegroundColor Gray
Write-Host ""

# 运行推理
Invoke-Expression $cmd

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "推理完成！" -ForegroundColor Green
Write-Host "结果保存在: $OUTPUT" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

