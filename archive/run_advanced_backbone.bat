@echo off
REM Windows批处理脚本 - 使用最强backbone（EfficientNet-B7）
python zero_shot_inference_advanced.py --data_root_train MMAR/train_500 --video_list_train MMAR/train_500/train_videofolder_500.txt --data_root_test MMAR/test_200 --video_list_test MMAR/test_200/test_videofolder_200.txt --model_name efficientnet_b7 --batch_size 1 --classifier logistic --output submission.csv --gpu 0
pause

