# 基于92.54分的保守优化版本（目标95+分）
# 策略：只增加TTA次数，其他配置与92.54分完全一致

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "95+分保守优化版本" -ForegroundColor Cyan
Write-Host "只增加TTA次数，其他配置与92.54分完全一致" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 尝试不同的TTA次数
Write-Host "尝试TTA=20次..." -ForegroundColor Green
python fine_tune_convnext_95plus_conservative.py `
    --use_pose `
    --tta_times 20 `
    --classifier ensemble `
    --output submission_95plus_tta20.csv `
    --gpu 0

Write-Host ""
Write-Host "完成！" -ForegroundColor Green
Write-Host ""
Write-Host "如果还没到95+分，可以尝试：" -ForegroundColor Yellow
Write-Host "  --tta_times 25  (更激进)" -ForegroundColor Yellow
Write-Host "  --tta_times 30  (非常激进)" -ForegroundColor Yellow




