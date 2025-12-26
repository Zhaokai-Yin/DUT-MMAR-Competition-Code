@echo off
REM Windows批处理脚本 - 零样本推理
python zero_shot_inference.py --data_root_train MMAR/train_500 --video_list_train MMAR/train_500/train_videofolder_500.txt --data_root_test MMAR/test_200 --video_list_test MMAR/test_200/test_videofolder_200.txt --batch_size 2 --model_name r2plus1d_18 --classifier logistic --output submission.csv --gpu 0
pause




