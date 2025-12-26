# ConvNeXt V2 Large版本（目标95+分）
# 策略：明确使用ConvNeXt V2 Large，其他配置与92.54分完全一致

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ConvNeXt V2 Large版本（目标95+分）" -ForegroundColor Cyan
Write-Host "其他配置与92.54分完全一致" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python fine_tune_convnext_v2_95plus.py `
    --use_pose `
    --tta_times 10 `
    --classifier ensemble `
    --output submission_convnext_v2_95plus.csv `
    --gpu 0

Write-Host ""
Write-Host "完成！" -ForegroundColor Green



