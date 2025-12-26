# 基于92.54分的最终优化版本（目标95+分）
# 策略：只微调分类器参数，不改变特征提取
# 如果这个还不行，可能需要模型微调或其他更复杂的方法

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "最终优化版本（目标95+分）" -ForegroundColor Cyan
Write-Host "策略：只微调分类器参数" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python fine_tune_convnext_95plus_final.py `
    --use_pose `
    --tta_times 10 `
    --classifier ensemble `
    --output submission_95plus_final.csv `
    --gpu 0

Write-Host ""
Write-Host "完成！" -ForegroundColor Green

