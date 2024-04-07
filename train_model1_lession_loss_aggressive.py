# https://youtu.be/ScdCQqLtnis
"""
@author: Sreenivas Bhattiprolu

Code to train batches of cropped BraTS 2020 images using 3D U-net.

Please get the data ready and define custom data gnerator using the other
files in this directory.

Images are expected to be 64x64x64x3 npy data (3 corresponds to the 3 channels for 
                                                  test_image_flair, test_image_t1ce, test_image_t2)
Change the U-net input shape based on your input dataset shape (e.g. if you decide to only se 2 channels or all 4 channels)

Masks are expected to be 64x64x64x3 npy data (4 corresponds to the 4 classes / labels)


You can change input image sizes to customize for your computing resources.
"""


import os
import numpy as np
import tensorflow as tf
#import tensorflow.keras as keras
import keras
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
import glob
import random
from scipy.spatial.distance import directed_hausdorff

def load_img(img_list):
    images=[]
    msks=[]
    for i, image_name in enumerate(img_list):
        image = np.load(image_name)
        images.append(image)
        msk_name=image_name.replace("images", "masks")
        msk_name=msk_name.replace("image", "mask")
        msk= np.load(msk_name)
        msk[msk>0] = 1

        msks.append(np.eye(2)[msk])
        #msks.append(msk)
        print(f"loading image {image_name}")

    images, masks = np.array(images), np.array(msks)
    #img_tensor = tf.convert_to_tensor(images)
    return images, masks

def imageLoader(img_dir,  batch_size, filesize):
    #img2_list = glob.glob("grid_data/non_cancer/"+img_dir+"/*.npy")
    img1_list = glob.glob("grid_data/cancer/"+img_dir+"/*.npy")
    fsize = filesize -len(img1_list)
    while True:
        #random.shuffle(img2_list)
        img_list = img1_list #+ img2_list[:fsize]
        random.shuffle(img_list)

        batch_start = 0
        batch_end = batch_size
        L = len(img_list)

        while batch_end < L:
            limit = min(batch_end, L)

            X, Y = load_img(img_list[batch_start:limit])

            yield (X,Y)

            batch_start += batch_size
            batch_end += batch_size

# Check GPU availability
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

with tf.device('/GPU:0'):


    ####################################################

    #############################################################
    #Optional step of finding the distribution of each class and calculating appropriate weights
    #Alternatively you can just assign equal weights and see how well the model performs: 0.25, 0.25, 0.25, 0.25

    import pandas as pd
    #columns = ['0', '1', '2', '3']
    #df = pd.DataFrame(columns=columns)
    #train_mask_list = sorted(glob.glob('masks_train/*.npy'))
    ##for img in range(len(train_mask_list)):
    #    print(img)
    #    temp_image=np.load(train_mask_list[img])
    #   temp_image = np.argmax(temp_image, axis=3)
    #    val, counts = np.unique(temp_image, return_counts=True)
    #    zipped = zip(columns, counts)
    #   conts_dict = dict(zipped)
        
    #    df = df.append(conts_dict, ignore_index=True)

    #label_0 = df['0'].sum()
    #label_1 = df['1'].sum()
    #label_2 = df['1'].sum()
    #label_3 = df['3'].sum()
    #total_labels = label_0 + label_1 + label_2 + label_3
    #n_classes = 4
    #Class weights claculation: n_samples / (n_classes * n_samples_for_class)
    #wt0 = round((total_labels/(n_classes*label_0)), 2) #round to 2 decimals
    #wt1 = round((total_labels/(n_classes*label_1)), 2)
    #wt2 = round((total_labels/(n_classes*label_2)), 2)
    #wt3 = round((total_labels/(n_classes*label_3)), 2)

    #Weights are: 0.26, 22.53, 22.53, 26.21
    #wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21
    #These weihts can be used for Dice loss 

    ##############################################################
    #Define the image generators for training and validation

    train_img_dir = "images_train/"
    train_mask_dir = "masks_train/"

    val_img_dir = "images_validate/"
    val_mask_dir = "masks_validate/"

    ##################################

    ########################################################################
    batch_size = 16
    train_img_list = 1* len(glob.glob(f"grid_data/cancer/images_train/*.npy"))
    val_img_list = 1 * len(glob.glob(f"grid_data/cancer/images_validate/*.npy"))

    train_img_datagen = imageLoader(train_img_dir, batch_size, train_img_list)
    val_img_datagen = imageLoader(val_img_dir,  batch_size, val_img_list)

    # #Verify generator.... In python 3 next() is renamed as __next__()
    test_img, test_mask = train_img_datagen.__next__()
    sum_xy = np.sum(np.argmax(test_mask[0], axis=3), axis=(0, 1))
    n_slice = np.argmax(sum_xy)
    print (n_slice)
    plt.figure(figsize=(24,16))

    plt.subplot(221)
    plt.imshow(test_img[0,:, :,n_slice])
    plt.title('Image flair')
    plt.subplot(224)
    plt.imshow(np.argmax(test_mask[0,:,:,n_slice], axis=2))
    plt.title('Mask')
    plt.show()


    # Compute metric between the predicted segmentation and the ground truth
   #Keras

    def DiceCoe0(targets, inputs, smooth=1e-6):
        dice = 0.0
        input = K.flatten(inputs[:,:,:,:,0])
        target = K.flatten(targets[:,:,:,:, 0])
        intersection = K.sum(target * input)
        dice += (2*intersection + smooth) / (K.sum(target) + K.sum(input) + smooth)
        return dice


    def DiceCoe1(targets, inputs, smooth=1e-6):
        dice = 0.0
        input = K.flatten(inputs[:,:,:,:,1])
        target = K.flatten(targets[:,:,:,:, 1])
        intersection = K.sum(target * input)
        dice += (2*intersection + smooth) / (K.sum(target) + K.sum(input) + smooth)
        return dice


    def FalsePositive(targets, inputs, smooth=1e-6):
        dice = 0.0
        input = K.flatten(inputs[ :,:,:,:,0])
        target = K.flatten(targets[:,:,:,:, 0])
        intersection = K.sum(target * input)

        dice += (intersection+smooth) / (K.sum(targets[:,:,:,:, 0]) + smooth)
        return 1-dice

    def FalseNegative(targets, inputs, smooth=1e-6):
        dice = 0.0
        input = K.flatten(inputs[ :,:,:,:,1])
        target = K.flatten(targets[:,:,:,:, 1])
        intersection = K.sum(target * input)

        dice += (intersection+smooth) / (K.sum(targets[:,:,:,:, 1]) + smooth)
        return 1- dice

    def LessionDetectionLoss(targets, inputs, smooth=1e-6):
        found = 0.0
        target = 0.0
        for i in range(batch_size):
            if K.sum(K.flatten(targets[ i,:,:,:,1])) > 100:
                target += 1.0
            if K.sum(K.flatten(inputs[ i,:,:,:,1])) <= 100 and K.sum(K.flatten(targets[ i,:,:,:,1])) > 100:
                found += 1.0
        return  (found)/(found + target+smooth)
        
    def LessionFalseLoss(targets, inputs, smooth=1e-6):
        found = 0.0
        target = 0.0
        for i in range(batch_size):
            if K.sum(K.flatten(targets[ i,:,:,:,1])) <= 100 and K.sum(K.flatten(inputs[ i,:,:,:,1])) > 100 :
                found += 1.0
            if K.sum(K.flatten(targets[ i,:,:,:,1])) <= 100:
                target += 1.0

        return  (found)/(target+found+smooth)

    def LessionAccuracy(targets, inputs, smooth=1e-6):
        found = 0.0
        target = 0.0
        for i in range(batch_size):
            if K.sum(K.flatten(targets[ i,:,:,:,1])) > 100 and K.sum(K.flatten(inputs[ i,:,:,:,1])) > 100 :
                target += 1.0
            if K.sum(K.flatten(targets[ i,:,:,:,1])) <= 100 and K.sum(K.flatten(inputs[ i,:,:,:,1])) <= 100 :
                target += 1.0
        return  (target+smooth)/(batch_size+smooth)


    def DiceLoss(targets, inputs, smooth=1e-6):
        wt0, wt1 = 1,  10.0
        #flatten label and prediction tensors
        dice = wt0*DiceCoe0(targets, inputs) +  wt1*DiceCoe1(targets, inputs) #+  wt2*DiceCoe2(targets, inputs)
        return 11 - dice

    ALPHA = 0.8
    GAMMA = 2

    def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
        
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        BCE = K.binary_crossentropy(targets, inputs)
        BCE_EXP = K.exp(-BCE)
        focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
        
        return focal_loss



    def TotalLoss(targets, inputs):    
        return   FocalLoss(targets, inputs) + DiceLoss(targets, inputs) + LessionDetectionLoss(targets, inputs) + LessionFalseLoss(targets, inputs) 

    #Define loss, metrics and optimizer to be used for training

    metrics = ['accuracy', FocalLoss, DiceCoe0, DiceCoe1, FalsePositive,  FalseNegative,  LessionAccuracy,  LessionDetectionLoss,  LessionFalseLoss, keras.metrics.Precision(thresholds=0)]

    LR = 0.000075
    #LR = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=.0001, decay_steps=20, decay_rate=.1)
    optim = keras.optimizers.Adam(LR)
    #######################################################################
    #Fit the model 

    steps_per_epoch = train_img_list//batch_size
    val_steps_per_epoch = val_img_list //batch_size


    from simple_3d_unet import simple_unet_model

    model = simple_unet_model(IMG_HEIGHT=64, 
                            IMG_WIDTH=64,
                            IMG_DEPTH=64, 
                            IMG_CHANNELS=1, 
                            num_classes=2)

    from tensorflow.keras.callbacks import ModelCheckpoint

    # Assuming 'model', 'x_train', 'y_train', 'x_val', 'y_val' are defined and compiled
    # Define the ModelCheckpoint callback to save the best model based on validation accuracy
    checkpoint_path = 'saved_models/segmentation_grid_data_collect_cancer_only.hdf5'
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        monitor='loss',  # Metric to monitor
                                        save_best_only=True,     # Save only the best model
                                        mode='min',              # 'max' if maximizing the metric
                                        verbose=1)


    model.compile(optimizer=optim, loss=TotalLoss, metrics=metrics, run_eagerly=True)

    print(model.summary())

    print(model.input_shape)
    print(model.output_shape)

    history=model.fit(train_img_datagen,
            steps_per_epoch=steps_per_epoch,
            epochs=300,
            verbose=1,
            validation_data=val_img_datagen,
            validation_steps=val_steps_per_epoch,
            callbacks=[checkpoint_callback]
            )

    ##################################################################


    #plot the training and validation IoU and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()