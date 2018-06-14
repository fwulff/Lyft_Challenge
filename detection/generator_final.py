#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def fixed_crop_image(image, ground_truth1, ground_truth2, crop_top, crop_bottom):
    
    rows, cols, ch = image.shape  
    
    #crop
    image = image[crop_top:rows-crop_bottom, :, :]
    ground_truth1 = ground_truth1[crop_top:rows-crop_bottom, :]
    ground_truth2 = ground_truth2[crop_top:rows-crop_bottom, :]
    
    return image, ground_truth1, ground_truth2

def process_image(image, ground_truth1, ground_truth2, row, col, ch):
    
    #crop hood 520px - 600px = -80xp
    #crop sky 0 - 120px = -120px
    image, mask1, mask2 = fixed_crop_image(image, ground_truth1, ground_truth2, 120, 80)
    #print(image.shape)

    image = cv2.resize(image,(col, row))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    
    mask1 = cv2.resize(mask1, (col, row))
    mask1 = mask1.astype(np.float32)/255.
    mask1 = np.reshape(mask1, (row, col, 1))
    
    #print(ground_truth2.shape)
    mask2 = cv2.resize(mask2, (col, row))
    mask2 = mask2.astype(np.float32)/255.
    mask2 = np.reshape(mask2, (row, col, 1))
    
    return image, mask1, mask2

def standardize(image):

    norm_image = (image.astype(np.float32)/255.0)
    
    meanStd_array = cv2.meanStdDev(norm_image)

    norm_image[:,:,0] = (norm_image[:,:,0] - meanStd_array[0][0])/(meanStd_array[1][0])
    norm_image[:,:,1] = (norm_image[:,:,1] - meanStd_array[0][1])/(meanStd_array[1][1])
    norm_image[:,:,2] = (norm_image[:,:,2] - meanStd_array[0][2])/(meanStd_array[1][2])

    image = norm_image
        
    return image
    
def generate_batch(data1, data2, batch_size, ch, row, col):
    
    standardize_images = True
    
    if(standardize_images):
        image_batch = np.zeros((batch_size, row, col, ch)).astype(np.float32)
    else:
        image_batch = np.zeros((batch_size, row, col, ch)).astype(np.uint8)
        
    mask_batch = np.zeros((batch_size, row, col, 2)).astype(np.float32)
    
    while True: 
        
        # Shuffle again to reduce time series issues
        data1.iloc[np.random.permutation(len(data1))]
        data1.reset_index(drop=True)  

        # Shuffle again to reduce time series issues
        data2.iloc[np.random.permutation(len(data2))]
        data2.reset_index(drop=True)  
        
        for i_batch in range(batch_size):
            
            i_line = np.random.randint(len(data1))
            
            images = data1.iloc[i_line]['images'].strip()                 
            image = cv2.imread('./data_road/' + images)  
            
            mask = data1.iloc[i_line]['masks'].strip()
            #mask2 = data2.iloc[i_line]['masks'].strip()

            #read as greyscale
            mask1 = cv2.imread('./data_road/' + mask, 0)
            mask2 = cv2.imread('./data_car/' + mask, 0)

            #apply preprocessing
            image, mask1, mask2 = process_image(image, mask1, mask2, row, col, ch)
            
            if(standardize_images):
                image = standardize(image)
            
            #stack, concatenate?
            mask = np.concatenate([mask1,mask2], axis = 2)
            #print(mask.shape)

            image_batch[i_batch] = image
            mask_batch[i_batch] = mask
             
        yield image_batch, mask_batch