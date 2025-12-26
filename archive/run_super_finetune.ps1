# PowerShell脚本 - 超级强化快速微调（15个epoch）
python super_finetune.py `
    --data_root MMAR/train_500 `
    --video_list MMAR/train_500/train_videofolder_500.txt `
    --data_root_test MMAR/test_200 `
    --video_list_test MMAR/test_200/test_videofolder_200.txt `
    --backbone efficientnet `
    --model_size medium `
    --batch_size 2 `
    --accumulation_steps 2 `
    --epochs 15 `
    --lr 0.0003 `
    --fusion_method attention `
    --use_amp `
    --num_workers 2 `
    --gpu 0

