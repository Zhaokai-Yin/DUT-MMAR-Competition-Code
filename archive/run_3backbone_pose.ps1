# 3个backbone + 骨骼点配置
# 预期时间：2-3小时
# 预期分数：95-97分

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "3个Backbone + 骨骼点检测配置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置信息:" -ForegroundColor Yellow
Write-Host "  - Backbone: ConvNeXt Large + RegNet + EfficientNet V2" -ForegroundColor White
Write-Host "  - 骨骼点检测: 开启 (MediaPipe)" -ForegroundColor White
Write-Host "  - 分类器: 集成学习 (Logistic + SVM + RF)" -ForegroundColor White
Write-Host "  - TTA: 5次" -ForegroundColor White
Write-Host "  - 超参数优化: 开启" -ForegroundColor White
Write-Host ""
Write-Host "预期:" -ForegroundColor Yellow
Write-Host "  - 时间: 2-3小时" -ForegroundColor White
Write-Host "  - 分数: 95-97分" -ForegroundColor Green
Write-Host ""

$gpu = Read-Host "请输入GPU编号 (默认: 0)"
if ([string]::IsNullOrWhiteSpace($gpu)) {
    $gpu = 0
}

$output = Read-Host "输出文件名 (默认: submission_3backbone_pose.csv)"
if ([string]::IsNullOrWhiteSpace($output)) {
    $output = "submission_3backbone_pose.csv"
}

Write-Host "`n开始运行..." -ForegroundColor Green
Write-Host "提示: 程序会每50个batch自动保存检查点" -ForegroundColor Yellow
Write-Host "如果中断，可以从检查点恢复" -ForegroundColor Yellow
Write-Host ""

python zero_shot_inference_pose_enhanced.py `
    --backbone_names convnext_large,regnet,efficientnet_v2 `
    --use_pose `
    --classifier ensemble `
    --ensemble_models logistic,svm,rf `
    --tune_params `
    --tta_times 5 `
    --skip_cv `
    --save_checkpoint "$($output.Replace('.csv', '_checkpoint'))" `
    --checkpoint_interval 50 `
    --output $output `
    --gpu $gpu

Write-Host "`n完成！" -ForegroundColor Green

