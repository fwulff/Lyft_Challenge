
# Lyft Challenge

## Semantic segmentation of car and road pixel using FCNS
​

## Writeup

In this project, a fully convolutional networks is used to label the pixels of a road and cars in images.

​

### Dataset

​

* The network was trained on an extended dataset created with the CARLA simulator. The dataset has >5000 images and was contributed by other participants [Link](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz)

* The paths and filesnames of all images are saved in a .csv file, which is parsed by the generator.

* The ground truth is preprocessed into two separate binary images, where ones label the road / car pixels and zeros label the background / ignore label.

* The network therefore predicts a two-channel image, where each channel is encoding a class using a sigmoid function for binary values

* The hood is removed from the car class pixelwise (not cropped) using images from the ground truth, where no car was present and subtracting the _car_ class

​

### FCN Network

​

A Fully Convolutional Network (FCN) is trained to label each pixel into 2 classes (each binary) - this can be done by the following three approaches:

​

* 1x1 convolutions

* Skip layers to concatenate low level and high level features / convolutions

* Upsampling oder Deconvolution using transposed convolution to restore the original input image size

​

A pretrained VGG19 network is used as the encoder, where all fully connected layers are removed and upsampling and skip connections are added.

​

The creates a pre-trained fully-convolutional network, as described as the _UNET_ in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

​

```

#Define network - UNET-FCN with VGG19 Layers and pre-trained weights

def FCN_Vgg19_unet(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=1):

​

    # Image Input

    ImageInput = Input(shape=(row, col, ch), name = 'image')

​

    # Block 1

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(ImageInput)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(conv1)

    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

​

    # Block 2

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(pool1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(conv2)

    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

​

    # Block 3

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(pool2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(conv3)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(conv3)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', kernel_regularizer=l2(weight_decay))(conv3)

    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

​

    # Block 4

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(pool3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(conv4)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(conv4)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', kernel_regularizer=l2(weight_decay))(conv4)

    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

​

    # Block 5

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(conv5)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(conv5)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4', kernel_regularizer=l2(weight_decay))(conv5)

​

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(up6)

    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(conv6)

    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(conv6)

    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv4')(conv6)

    

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv1')(up7)

    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv2')(conv7)

    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv3')(conv7)

    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv4')(conv7)

    

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv1')(up8)

    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv2')(conv8)

​

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv1')(up9)

    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv')(conv9)

    

    output = Conv2D(2, (1, 1), name="ground_truth_2", activation='sigmoid')(conv9)

    

    model = Model(inputs=ImageInput, outputs=output)

​

    model.compile(loss=IOU_multi_loss, optimizer=Adam(lr = learning_rate))

    

    #load pretrained weights from keras 

    #https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5

    #weights_path = './model_weights/vgg19_weights_tf_dim_ordering_tf_kernels.h5'

    #model.load_weights(weights_path, by_name=True)

    

    # Load trained weights - if resuming training

    model_weights = "./model_weights/Best_FCN_Vgg19_unet_final.hdf5"

    model.load_weights(model_weights)

    

    print("Weights loaded")    

​

    return model

```

​

### Optimizer and loss function

​

* As the optimizer, ADAM is used with a learning rate of 1e-5. 

* The loss is calculated using a custom Intersection-over-Union (IOU) using binary True/False values. The IOU is implemented using Keras syntax to be run and compiled with the network on the GPU.

* As the network is trained to predict two channels, the IOU is calculated separately for the road and car binary prediction and added weighted. (cars are weighted 10 times, as the network struggles to predict cars)

​

```

#Define IOU coefficient as loss function for a binary image

def IOU_calc(y_true, y_pred):

    y_true_f = K.batch_flatten(y_true)

    y_pred_f = K.batch_flatten(y_pred)

    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth

    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth

    return K.mean(intersection / union)

​

def IOU_calc_loss(y_true, y_pred):

    return -IOU_calc(y_true, y_pred)

​

#average IOU loss for multiple binary images - channel 0:road - channel 1:cars

def IOU_multi_loss(y_true, y_pred):

    iou = 0.

    iou -= IOU_calc(y_true[:,:,:,0], y_pred[:,:,:,0])

    iou -= 9*IOU_calc(y_true[:,:,:,1], y_pred[:,:,:,1])

    return (iou/10.)

```

​

### Preprocessing

​

A generator is used to yield batches, reduces the required memory and apply preprocessing and image augmentations.

​

Images are preprocessed:

* Normalized using mean and stddev on each channel of the image separately

* YUV colorspace to avoid correlation between channels (as for intensity in RGB images)

* Cropped to remove lower part (hood) and upper part (sky), where car or road class is never present

​

### Performance

​

To increase the performance to approx. 10 FPS, the following approaches are taken:

* resolution of 304 x 144 px

* frozen .pb tensorflow graph instead of Keras

* optimized for inference graph

* using only OpenCV in Python for image and video-processing, as it far outperforms PIL and Skvideo (cv2 is compiled C++ code called in Python)

​

### Training

​

Training is done for :

* 30 epochs

* A resolution of 576 x 160 px 

* Learning rate of 1e-5

* Batch size of 16 (Tesla K80 GPU with 12GB RAM used)

* Checkpoints are used to save the best performing model, not the final one

* Early stopping is used to prevent overfitting, if validation loss is not increasing anymore

​

```

# create checkpoints for each improvement

# use IOU as val_loss

checkpoint = ModelCheckpoint(filepath = './model_weights/Best_'+model_weights, verbose = 1, save_best_only=True, monitor='val_loss')

​

# early termination with epochs as patience = 2 to prevent overfitting

callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

​

# fits the model on batches with real-time data augmentation:

# https://keras.io/models/model/

history = model.fit_generator(train_data_gen,

                    steps_per_epoch = set_steps_per_epoch(data_size_train_road, batch_size), 

                    epochs = epochs, 

                    verbose = 1, 

                    callbacks = [checkpoint, callback], 

                    validation_data = val_data_gen,

                    validation_steps = set_steps_per_epoch(data_size_val_road, batch_size))

```

​

## Results

​

* The training loss is decreasing continously to a value of approx. -0.90. 

* The resulting road segmentation is showing excellent results, labelling a majority > 95% of road pixels correct. The architecture and training was optimized for a road detection in the Master Thesis and Paper previously.

* The resulting car segmentation is still underperforming. Due to the small size of the cars and the rare presence, the network trains far worse on cars. Image augmentation could help to improve significantly, cropping and zooming cars frequently for example.

* The road segmentation reached an fScore of 0.965 in the grader.

* The car segmentation reached an fScore of 0.536 in the grader.

* The resulting score is 0.750

​

```

Your program runs at 9.523 FPS

​

Car F score: 0.536 | Car Precision: 0.884 | Car Recall: 0.488 | Road F score: 0.965 | Road Precision: 0.997 | Road Recall: 0.855 | Averaged F score: 0.750

```

​

Anyway, this is still far away from the current state of the art benchmark. More recent architecture (VGG and UNET was introduced in 2015), higher resolutions, image augmentation and training optimization would be promising to increase the score.

​

