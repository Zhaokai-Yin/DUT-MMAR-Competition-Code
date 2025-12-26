# 零样本优化版本（目标95+分，不训练）
# 策略：多次运行取平均 + 强TTA + 优化温度缩放

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "零样本优化版本（目标95+分）" -ForegroundColor Cyan
Write-Host "不训练，只优化预测策略" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python fine_tune_convnext_95plus_zeroshot.py `
    --use_pose `
    --tta_times 20 `
    --num_runs 3 `
    --temperature 0.90 `
    --classifier ensemble `
    --output submission_95plus_zeroshot.csv `
    --gpu 0

Write-Host ""
Write-Host "完成！" -ForegroundColor Green



