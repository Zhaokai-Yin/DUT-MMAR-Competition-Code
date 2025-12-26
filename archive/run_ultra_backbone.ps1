# PowerShell脚本 - 使用最强ViT模型（ViT-Large）
python zero_shot_inference_ultra.py `
    --data_root_train MMAR/train_500 `
    --video_list_train MMAR/train_500/train_videofolder_500.txt `
    --data_root_test MMAR/test_200 `
    --video_list_test MMAR/test_200/test_videofolder_200.txt `
    --model_name vit_large_patch16_224 `
    --batch_size 1 `
    --classifier svm `
    --output submission.csv `
    --gpu 0
