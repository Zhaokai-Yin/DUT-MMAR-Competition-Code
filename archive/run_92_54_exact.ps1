# 精确复现92.54分的配置
# ConvNeXt-Large + 骨骼点 + SVM

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "精确复现92.54分配置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python fine_tune_convnext_92_54_exact.py `
    --use_pose `
    --tta_times 10 `
    --classifier ensemble `
    --output submission_92_54_exact.csv `
    --gpu 0

Write-Host ""
Write-Host "完成！" -ForegroundColor Green

