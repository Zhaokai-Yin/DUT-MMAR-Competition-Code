#!/bin/bash
python zero_shot_inference_enhanced.py --data_root_train MMAR/train_500 --video_list_train MMAR/train_500/train_videofolder_500.txt --data_root_test MMAR/test_200 --video_list_test MMAR/test_200/test_videofolder_200.txt --batch_size 2 --model_name r2plus1d_18 --classifier ensemble --ensemble_models logistic svm rf --use_pca --pca_components 512 --output submission_enhanced.csv --gpu 0








