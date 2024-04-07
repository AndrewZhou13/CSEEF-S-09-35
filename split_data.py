
import os
import shutil
import numpy as np
import tensorflow as tf
#import tensorflow.keras as keras
import keras
from keras import backend as K
from matplotlib import pyplot as plt
import glob
import random
from scipy.spatial.distance import directed_hausdorff


def load_img(img_list, a):
    images=[]
    for i, image_name in enumerate(img_list):
        x  = a[i]
        image_move_name = image_name.replace("validate", "train")
        if x < 0.7:
            shutil.move(image_name, image_move_name)
        print(f"moving image {image_name}")
    return images

def load_msk(img_list):
    a, b = [], []
    for i, msk_name in enumerate(img_list):
        msk_name=msk_name.replace("images", "masks")
        msk_name=msk_name.replace("image", "mask")
        msk = np.load(msk_name)
        msk_move_name = msk_name.replace("validate", "train")
        x = random.random()
        if x < .7:
            shutil.move(msk_name, msk_move_name)

        a.append(x)

        print(f"moving mask {msk_name}")
    #msk_tensor = tf.convert_to_tensor(images)
    return a, np.array(b)


img_list = glob.glob("grid_data/cancer/images_validate/*.npy")
a, b = load_msk(img_list)
print(np.histogram(b, density=True))
X = load_img(img_list, a)
plt.hist(b, bins = [0,50,100,200,400, 800,1600,3200, 6400, 12800, 25600, 100000]) 
plt.title("histogram") 
plt.show()

