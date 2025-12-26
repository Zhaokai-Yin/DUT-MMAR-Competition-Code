# 3个backbone + 骨骼点 - 快速版本（不调参）
# 预期时间：1.5-2小时
# 预期分数：94-96分

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "3个Backbone + 骨骼点 - 快速版本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置信息:" -ForegroundColor Yellow
Write-Host "  - Backbone: ConvNeXt Large + RegNet + EfficientNet V2" -ForegroundColor White
Write-Host "  - 骨骼点检测: 开启" -ForegroundColor White
Write-Host "  - 分类器: 集成学习 (Logistic + SVM + RF)" -ForegroundColor White
Write-Host "  - 超参数调优: 关闭（使用默认参数）" -ForegroundColor White
Write-Host "  - TTA: 5次" -ForegroundColor White
Write-Host "  - 交叉验证: 跳过" -ForegroundColor White
Write-Host ""
Write-Host "预期:" -ForegroundColor Yellow
Write-Host "  - 时间: 1.5-2小时" -ForegroundColor White
Write-Host "  - 分数: 94-96分" -ForegroundColor Green
Write-Host ""

$gpu = Read-Host "请输入GPU编号 (默认: 0)"
if ([string]::IsNullOrWhiteSpace($gpu)) {
    $gpu = 0
}

$output = Read-Host "输出文件名 (默认: submission_3backbone_pose_fast.csv)"
if ([string]::IsNullOrWhiteSpace($output)) {
    $output = "submission_3backbone_pose_fast.csv"
}

Write-Host "`n开始运行..." -ForegroundColor Green
Write-Host "提示: 不进行超参数调优，速度更快" -ForegroundColor Yellow
Write-Host ""

python zero_shot_inference_pose_enhanced.py `
    --backbone_names convnext_large,regnet,efficientnet_v2 `
    --use_pose `
    --classifier ensemble `
    --ensemble_models logistic,svm,rf `
    --skip_cv `
    --tta_times 5 `
    --checkpoint_interval 50 `
    --output $output `
    --gpu $gpu

Write-Host "`n完成！" -ForegroundColor Green




