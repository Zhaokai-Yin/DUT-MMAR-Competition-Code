# PowerShell脚本 - 超级无敌爆炸厉害版本（集成多个最强模型）
python zero_shot_inference_super.py `
    --data_root_train MMAR/train_500 `
    --video_list_train MMAR/train_500/train_videofolder_500.txt `
    --data_root_test MMAR/test_200 `
    --video_list_test MMAR/test_200/test_videofolder_200.txt `
    --models vit_l_16,convnext_large,efficientnet_b7 `
    --batch_size 1 `
    --classifier svm `
    --output submission.csv `
    --gpu 0

