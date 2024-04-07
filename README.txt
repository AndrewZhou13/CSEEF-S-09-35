get_image_file:   code to convert .nii input into .npy numpy array
process_data_128.py:  code to split input data into uniform 64x64x64 blocks
split_data.py:  code to split data into 70% train and 30% validate
simple_3d_unet.py:  tensorflow model for 3d U-Net model
train_model1_lession_loss_aggressive.py: input python code to train aggresssive model
train_model1_lession_loss_passive.py: input python code to train passive model
train_model1_lession_loss_aggressive.txt:  output log file for training aggresssive model

train_model1_data_collect_cancer_only.py:  data collection python code to calculate statistics with aggresive model
train_model1_data_collect_cancer_only.txt: data collection output from train_model1_data_collect_cancer_only.py for 50 Epoch (zipped)
train model1 aggressive model stats.csv:  analysis of data collected from train_model1_data_collect_cancer_only.txt,  mean and std deviation reported

train_model1_data_collect_cancer_noncancer_mix.py:    data collection python code to calculate statistics with pasive model
train_model1_data_collect_cancer_noncancer_mix.txt: data collection output from train_model1_data_collect_cancer_noncancer_mix.py for 50 Epoch (zippped)
train model1 passive model stats.csv:  analysis of data collected from train_model1_data_collect_cancer_noncancer_mix.txt,  mean and std deviation reported

train_model1_predict_msks_cancer.out:  output file for predicting mask for cancer blocks using aggressive model (zipped)
train_model1_predict_msks_noncancer.out:  output file for predicting mask for non-cancer blocks using aggressive model (zipped)