import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import glob
import random
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def convert_to_npy(img_list):
    images=[]

    for i, image_name in enumerate(img_list):
        data = nib.load(image_name).get_fdata()
        data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        npy_name=image_name.replace(".nii", ".npy")
        print(image_name, data.shape)
        np.save(npy_name, data)


img_list = glob.glob("ucsf_raw_data/masks_validate/*.nii")
convert_to_npy(img_list)