# ConvNeXt Large快速微调脚本（目标95+分）
# 只训练8个epoch，预计1-2小时

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ConvNeXt Large快速微调（目标95+分）" -ForegroundColor Cyan
Write-Host "只训练8个epoch，预计1-2小时" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "步骤1: 训练模型（预计1-2小时）" -ForegroundColor Green
python train_convnext_large_fast.py `
    --data_root MMAR/train_500 `
    --video_list MMAR/train_500/train_videofolder_500.txt `
    --batch_size 1 `
    --accumulation_steps 4 `
    --epochs 8 `
    --lr 0.0005 `
    --fusion_method late `
    --use_amp `
    --save_dir checkpoints_convnext_large_fast `
    --gpu 0

Write-Host ""
Write-Host "步骤2: 生成提交文件" -ForegroundColor Green
python generate_submission_convnext_large.py `
    --checkpoint checkpoints_convnext_large_fast/best_model.pth `
    --tta_times 10 `
    --output submission_convnext_large_fast.csv `
    --gpu 0

Write-Host ""
Write-Host "完成！" -ForegroundColor Green



