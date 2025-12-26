# 基于92.54分的优化版本（目标95+分）
# 优化策略：
# 1. 增加TTA次数（15次）
# 2. 多时序聚合（mean, max, std）
# 3. 优化分类器参数
# 4. 优化温度缩放（0.92）

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "95+分优化版本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python fine_tune_convnext_95plus_optimized.py `
    --use_pose `
    --use_multi_temporal `
    --tta_times 15 `
    --temperature 0.92 `
    --classifier ensemble `
    --output submission_95plus_optimized.csv `
    --gpu 0

Write-Host ""
Write-Host "完成！" -ForegroundColor Green




