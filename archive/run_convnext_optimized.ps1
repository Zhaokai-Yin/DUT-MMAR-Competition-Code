# ConvNeXt Large 优化版本快速启动脚本
# 基于91.54分的基础配置，目标95分+

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ConvNeXt Large 优化版本" -ForegroundColor Cyan
Write-Host "基础分数: 91.54分" -ForegroundColor Yellow
Write-Host "目标分数: 95分+" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 选择配置
Write-Host "请选择配置:" -ForegroundColor Yellow
Write-Host "1. 快速配置（集成学习，TTA=5，无调参）- 预期94分，~1小时" -ForegroundColor White
Write-Host "2. 推荐配置（集成学习，TTA=5，有调参）- 预期94-95分，~1.5小时" -ForegroundColor White
Write-Host "3. 最强配置（集成学习，TTA=10，有调参）- 预期95-97分，~2小时" -ForegroundColor White
Write-Host "4. 自定义配置" -ForegroundColor White
Write-Host ""

$choice = Read-Host "请输入选项 (1-4)"

$gpu = Read-Host "请输入GPU编号 (默认: 0)"
if ([string]::IsNullOrWhiteSpace($gpu)) {
    $gpu = 0
}

switch ($choice) {
    "1" {
        Write-Host "`n使用快速配置..." -ForegroundColor Green
        python zero_shot_inference_convnext_optimized.py `
            --classifier ensemble `
            --ensemble_models logistic,svm,rf `
            --tta_times 5 `
            --output submission_convnext_fast.csv `
            --gpu $gpu
    }
    "2" {
        Write-Host "`n使用推荐配置..." -ForegroundColor Green
        python zero_shot_inference_convnext_optimized.py `
            --classifier ensemble `
            --ensemble_models logistic,svm,rf `
            --tune_params `
            --tta_times 5 `
            --output submission_convnext_recommended.csv `
            --gpu $gpu
    }
    "3" {
        Write-Host "`n使用最强配置..." -ForegroundColor Green
        python zero_shot_inference_convnext_optimized.py `
            --classifier ensemble `
            --ensemble_models logistic,svm,rf `
            --tune_params `
            --tta_times 10 `
            --output submission_convnext_95plus.csv `
            --gpu $gpu
    }
    "4" {
        Write-Host "`n自定义配置..." -ForegroundColor Green
        $classifier = Read-Host "分类器类型 (logistic/svm/rf/ensemble, 默认: ensemble)"
        if ([string]::IsNullOrWhiteSpace($classifier)) {
            $classifier = "ensemble"
        }
        
        $ensemble_models = Read-Host "集成模型列表 (用逗号分隔, 默认: logistic,svm,rf)"
        if ([string]::IsNullOrWhiteSpace($ensemble_models)) {
            $ensemble_models = "logistic,svm,rf"
        }
        
        $tta = Read-Host "TTA次数 (默认: 5)"
        if ([string]::IsNullOrWhiteSpace($tta)) {
            $tta = 5
        }
        
        $tune = Read-Host "是否调参? (y/n, 默认: n)"
        $tune_params = ""
        if ($tune -eq "y" -or $tune -eq "Y") {
            $tune_params = "--tune_params"
        }
        
        $output = Read-Host "输出文件名 (默认: submission_convnext_custom.csv)"
        if ([string]::IsNullOrWhiteSpace($output)) {
            $output = "submission_convnext_custom.csv"
        }
        
        $cmd = "python zero_shot_inference_convnext_optimized.py --classifier $classifier --ensemble_models $ensemble_models --tta_times $tta $tune_params --output $output --gpu $gpu"
        Invoke-Expression $cmd
    }
    default {
        Write-Host "无效选项，使用推荐配置..." -ForegroundColor Yellow
        python zero_shot_inference_convnext_optimized.py `
            --classifier ensemble `
            --ensemble_models logistic,svm,rf `
            --tune_params `
            --tta_times 5 `
            --output submission_convnext_recommended.csv `
            --gpu $gpu
    }
}

Write-Host "`n完成！" -ForegroundColor Green




