# 终极零样本优化版本（目标95+分，不训练）
# 策略：特征选择 + 概率校准 + 优化参数 + 优化温度缩放

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "终极零样本优化版本（目标95+分）" -ForegroundColor Cyan
Write-Host "不训练，尝试所有可能的优化策略" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python fine_tune_convnext_95plus_ultimate.py `
    --use_pose `
    --tta_times 10 `
    --use_feature_selection `
    --feature_selection_k 3000 `
    --use_calibration `
    --temperature 0.88 `
    --classifier ensemble `
    --output submission_95plus_ultimate.csv `
    --gpu 0

Write-Host ""
Write-Host "Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "If still not 95+, may need:" -ForegroundColor Yellow
Write-Host "  1. Model fine-tuning (requires training)" -ForegroundColor Yellow
Write-Host "  2. Or accept 92.54 is the limit of zero-shot method" -ForegroundColor Yellow

