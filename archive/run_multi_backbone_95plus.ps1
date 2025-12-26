# 基于92.54分的多backbone版本（目标95+分）
# 策略：使用多个backbone（ConvNeXt + RegNet + EfficientNet V2）+ 骨骼点
# 关键：使用L2归一化避免特征尺度差异

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "多backbone版本（目标95+分）" -ForegroundColor Cyan
Write-Host "Backbone: ConvNeXt + RegNet + EfficientNet V2" -ForegroundColor Yellow
Write-Host "关键：使用L2归一化" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python fine_tune_convnext_multi_backbone_95plus.py `
    --backbone_names convnext_large,regnet,efficientnet_v2 `
    --use_pose `
    --tta_times 10 `
    --classifier ensemble `
    --output submission_multi_backbone_95plus.csv `
    --gpu 0

Write-Host ""
Write-Host "完成！" -ForegroundColor Green



