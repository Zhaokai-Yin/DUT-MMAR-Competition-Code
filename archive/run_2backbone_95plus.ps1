# 基于92.54分的2-backbone版本（目标95+分）
# 策略：只使用2个backbone（ConvNeXt + RegNet），减少特征维度
# 关键：使用L2归一化，保持其他配置与92.54分一致

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "2-backbone版本（目标95+分）" -ForegroundColor Cyan
Write-Host "Backbone: ConvNeXt + RegNet" -ForegroundColor Yellow
Write-Host "关键：使用L2归一化" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python fine_tune_convnext_2backbone_95plus.py `
    --backbone_names convnext_large,regnet `
    --use_pose `
    --tta_times 10 `
    --classifier ensemble `
    --output submission_2backbone_95plus.csv `
    --gpu 0

Write-Host ""
Write-Host "完成！" -ForegroundColor Green



