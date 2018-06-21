import sys, json, base64
import numpy as np

#from PIL import Image
from io import BytesIO, StringIO

import tensorflow as tf

import json
import cv2

#ch, col, row = 3, 400, 192
ch, col, row = 3, 304, 144  # camera format

def load_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess, ops
    
def standardize(image):
        
        #range 0 ... 1.0
        norm_image = (image.astype(np.float32)/255.0)
        
        meanStd_array = cv2.meanStdDev(norm_image)

        norm_image[:,:,0] = (norm_image[:,:,0] - meanStd_array[0][0])/(meanStd_array[1][0])
        norm_image[:,:,1] = (norm_image[:,:,1] - meanStd_array[0][1])/(meanStd_array[1][1])
        norm_image[:,:,2] = (norm_image[:,:,2] - meanStd_array[0][2])/(meanStd_array[1][2])

        image = norm_image
        
        return image
    
def process_image(image, row, col, ch):

    image = image[180:520, :, :]
    #print(image.shape)
        
    image = cv2.resize(image,(col, row))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    image = standardize(image)
    
    # add channel
    image = image[np.newaxis, ...]
    
    return image

def load_model():
    
    sess, _ = load_graph('./optimized_graph.pb')
    image = sess.graph.get_tensor_by_name('image:0')
    sigmoid = sess.graph.get_tensor_by_name('ground_truth_2/Sigmoid:0')

    return sess, image, sigmoid
    
file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
  
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")

#increase FPS using cv2 instead of PIL and skvideo
video = cv2.VideoCapture(file)
    
answer_key = {}

# Frame numbering starts at 1
frame = 1

model, image, sigmoid = load_model()

#for rgb_frame in video:
while(video.isOpened()):
    
    ret, rgb_frame = video.read()
    
    if ret is True:
        
        #prediction
        img = process_image(rgb_frame, row, col, ch)
        prediction = model.run(sigmoid, {image: img})

        # Look for cars :)
        car_result = np.zeros((600,800))
        #channel one is cars
        car_prediction = np.reshape(prediction[:,:,:,1], (row, col, 1))
        car_prediction = cv2.resize(car_prediction , (800, 340))
        car_result[180:520,:] = car_prediction
        #car_result[(car_result[:,:] > 0.)] = 1.
        car_result = car_result*255
        #print(np.max(car_result))

        car_frame_name = str(frame) + "_car.png"
        cv2.imwrite(car_frame_name, car_result)

        # Look for road :)
        road_result = np.zeros((600,800))
        #channel zero is road
        road_prediction = np.reshape(prediction[:,:,:,0], (row, col, 1))
        road_prediction = cv2.resize(road_prediction , (800, 340))
        road_result[180:520,:] = road_prediction
        #road_result[(road_result[:,:] > 0.)] = 1.
        road_result = road_result*255

        #print(np.max(road_result))

        road_frame_name = str(frame) + "_road.png"
        cv2.imwrite(road_frame_name, road_result)

        overlay = np.zeros_like(rgb_frame)
        overlay[:,:,2] = car_result
        overlay[:,:,1] = road_result
    
        final_frame = cv2.addWeighted(rgb_frame, 1, overlay, 0.3, 0, rgb_frame)
        final_frame_name = str(frame) + ".png"
        cv2.imwrite(final_frame_name, final_frame)
        # Increment frame
        frame+=1
        
    #all frames are run
    else:
        video.release()