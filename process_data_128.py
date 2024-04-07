import os
import numpy as np
import tensorflow as tf
#import tensorflow.keras as keras
import keras
from keras import backend as K
from matplotlib import pyplot as plt
import glob
import random
from scipy.spatial.distance import directed_hausdorff


def load_img_msk(img_list):
    images=[]

    for i, image_name in enumerate(img_list):
        msk_name=image_name.replace("images", "masks")
        msk_name=image_name.replace("image", "mask")

        msk = np.load(msk_name)
        image = np.load(image_name)

        msk = np.argmax(msk, axis=3)
        msk[msk>0] = 1

        #padding up multiples of grid size
        nx, ny, nz = (msk.shape[0]+32)//64, (msk.shape[1]+32)//64, (msk.shape[2]+32)//64
        target_shape = (nx*64, ny*64, nz*64)
        print(image_name, nx, ny, nz)

        # Calculate the required padding for each dimension
        pad_width = [(max(0, (target_shape[i] - msk.shape[i]) // 2), 
                    max(0, (target_shape[i] - msk.shape[i]) // 2)) for i in range(msk.ndim)]
    
        # pad to fill up borders to integers of 64
        msk = np.pad(msk, pad_width=pad_width, mode='constant', constant_values=0)
        image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)

        # crop down if oversized
        x0, y0, z0 = (msk.shape[0] - nx*64)//2, (msk.shape[1] - ny*64)//2, (msk.shape[2] - nz*64)//2
        msk = msk[x0:x0+nx*64, y0:y0+ny*64, z0:z0+nz*64]
        image = image[x0:x0+nx*64, y0:y0+ny*64, z0:z0+nz*64]

        for i in range(0,nx):
            for j in range(0,ny):
                for k in range(nz):
                    msk_g = msk[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64]
                    img_g = image[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64]

                    if msk_g.sum() > 0:
                        save_dir = "grid_data/cancer"
                    else:
                        save_dir = "grid_data/non_cancer"

                    msk_save_name = msk_name.replace(".npy", f"_{i}_{j}_{k}.npy")
                    msk_save_name = msk_save_name.replace("raw_data", save_dir)
                    np.save(msk_save_name, msk_g)

                    img_save_name = image_name.replace(".npy", f"_{i}_{j}_{k}.npy")
                    img_save_name = img_save_name.replace("raw_data", save_dir)
                    np.save(img_save_name, img_g)

        for i in range(0,nx-1):
            for j in range(0,ny-1):
                for k in range(nz-1):
                    msk_g = msk[i*64+32:(i+1)*64+32, j*64+32:(j+1)*64+32, k*64+32:(k+1)*64+32]
                    img_g = image[i*64+32:(i+1)*64+32, j*64+32:(j+1)*64+32, k*64+32:(k+1)*64+32]

                    if msk_g.sum() > 0:
                        save_dir = "grid_data/cancer1"
                    else:
                        save_dir = "grid_data/cancer0"

                    msk_save_name = msk_name.replace(".npy", f"_{i}_{j}_{k}_h.npy")
                    msk_save_name = msk_save_name.replace("raw_data", save_dir)
                    np.save(msk_save_name, msk_g)

                    img_save_name = image_name.replace(".npy", f"_{i}_{j}_{k}_h.npy")
                    img_save_name = img_save_name.replace("raw_data", save_dir)
                    np.save(img_save_name, img_g)


img_list = glob.glob("raw_data/images_*/*T1*.npy")
load_img_msk(img_list)