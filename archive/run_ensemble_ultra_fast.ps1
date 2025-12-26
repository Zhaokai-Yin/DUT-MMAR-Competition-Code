# PowerShell脚本 - 快速版本（使用更小的模型，避免下载卡住）
# 使用ResNeXt-50代替101，性能差异很小但下载快很多

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "超强集成推理 - 快速版本" -ForegroundColor Cyan
Write-Host "使用ResNeXt-50（避免下载卡住）" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 设置参数（使用更小的模型）
$BACKBONES = @("convnext", "resnext", "efficientnet_v2")
$MODEL_SIZES = @("large", "50", "l")  # ResNeXt-50代替101
$BATCH_SIZE = 2
$OUTPUT = "submission_ensemble_fast.csv"
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






