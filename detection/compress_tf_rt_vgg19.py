import tensorflow as tf

import sys,os

import pandas as pd
import json

import numpy as np
import cv2

from keras.models import model_from_json
from keras import backend as K
from keras.optimizers import Adam, SGD

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
        return sess.graph, ops

# parse parameters
model_file = './model_weights/FCN_Vgg19_unet_final.json'
model_weights = './model_weights/Best_FCN_Vgg19_unet_final.hdf5'

# read in model
with open(model_file, 'r') as jfile:
    model = model_from_json(json.load(jfile))

print("Model loaded")    
        
# Load trained weights
model.load_weights(model_weights)
print("Model weights loaded")   

model.summary()

# Get model input and output names
model_input = model.input.name.strip(':0')
model_output = model.output.name.strip(':0')

print(model_input, model_output)

# Get session
sess = K.get_session()
graph = sess.graph.as_graph_def()

#print(graph)

print("got sess graph")

print(len(sess.graph.get_operations()))

#sess = tf.keras.backend.get_session()
#frozen_graph = tf.freeze_session(sess, output_names=[out.op.name for out in model.outputs])

# freeze graph and remove nodes used for training 
frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph, [model_output])
frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

#print("got frozen graph")

tf.train.write_graph(frozen_graph, "./", "frozen_graph.pb", as_text=False)
print("saved frozen_graph.pb file")

sess, frozen_graph = load_graph('frozen_graph.pb')
print(len(frozen_graph))

#run optimize for inference
#python -m tensorflow.python.tools.optimize_for_inference --input frozen_graph.pb --output optimized_graph.pb --input_names=image --output_names=ground_truth_2/Sigmoid