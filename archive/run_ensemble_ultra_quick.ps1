# PowerShell脚本 - 超快速版本（跳过Gradient Boosting）
# 如果Gradient Boosting训练太慢，使用这个版本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "超强集成推理 - 快速版本（跳过GB）" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 设置参数
$BACKBONES = @("convnext", "resnext", "efficientnet_v2")
$MODEL_SIZES = @("large", "50", "l")
$BATCH_SIZE = 2
$OUTPUT = "submission_ensemble_quick.csv"
$USE_TTA = $true
$SKIP_GB = $true  # 跳过Gradient Boosting

Write-Host "集成Backbone: $($BACKBONES -join ', ')" -ForegroundColor Yellow
Write-Host "模型大小: $($MODEL_SIZES -join ', ')" -ForegroundColor Yellow
Write-Host "批次大小: $BATCH_SIZE" -ForegroundColor Yellow
Write-Host "使用TTA: $USE_TTA" -ForegroundColor Yellow
Write-Host "跳过GB: $SKIP_GB" -ForegroundColor Yellow
Write-Host "输出文件: $OUTPUT" -ForegroundColor Yellow
Write-Host ""

# 构建命令
$cmd = "python zero_shot_inference_ensemble_ultra.py --backbones $($BACKBONES -join ' ') --model_sizes $($MODEL_SIZES -join ' ') --batch_size $BATCH_SIZE --output $OUTPUT --gpu 0"

if ($USE_TTA) {
    $cmd += " --use_tta"
}

if ($SKIP_GB) {
    $cmd += " --skip_gb"
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






