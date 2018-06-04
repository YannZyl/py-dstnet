# -*- coding: utf-8 -*-
import numpy as np
from resnet50 import ResNet50
from image import prepare_data
from image import data_loader
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import TimeDistributed
from tensorflow.contrib.keras.api.keras.layers import Bidirectional
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import Conv3D
from tensorflow.contrib.keras.api.keras.layers import Reshape
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.optimizers import SGD


def build_models(seq_len = 12, num_classes=4, load_weights=False):
    # DST-Net: ResNet50
    resnet = ResNet50(weights='imagenet', include_top=False)
    for layer in resnet.layers:
        layer.trainable = False
    resnet.load_weights('model/resnet.h5')
    # DST-Net: Conv3D + Bi-LSTM
    inputs = Input(shape=(seq_len, 7, 7, 2048))
    # conv1_1, conv3D and flatten
    conv1_1 = TimeDistributed(Conv2D(128, 1, 1, activation='relu'))(inputs)
    conv3d = Conv3D(64, 3, 1, 'SAME', activation='relu')(conv1_1) 
    flatten = Reshape(target_shape=(seq_len,7*7*64))(conv3d)
    # 2 Layers Bi-LSTM
    bilstm_1 = Bidirectional(LSTM(128,dropout=0.5, return_sequences=True))(flatten)
    bilstm_2 = Bidirectional(LSTM(128,dropout=0.5, return_sequences=False))(bilstm_1)
    outputs = Dense(num_classes, activation='softmax')(bilstm_2)
    dstnet = Model(inputs=inputs, outputs=outputs)
    dstnet.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))
    # load models
    if load_weights:
        dstnet.load_weights('model/dstnet.h5')
    return resnet, dstnet

def DSTNet_Extraction(resnet_model, frames):
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
    frames[...,0] -= 123.68
    frames[...,1] -= 116.779
    frames[...,2] -= 103.939
    features = resnet_model.predict(frames)
    return features

def DSTNet_Recognition(bilstm_model, sequence_feature):
    preds = bilstm_model.predict(sequence_feature)
    return np.argmax(preds, 1)

def train_script(video_dir='datas/videos', save_dir='datas/images'):
    prepare_data(video_dir, frame_gap=3, frames_per_group=12, save_dir='data_12')
    x, y = data_loader('data/', time_step=12)
    #x = np.random.randint(0, 256,(12,224,224,3)).astype(np.float32)
    #y = np.random.randint(0, 4, (16, 4)).astype(np.float32)
    resnet, dstnet = build_models()
    feats = np.array([DSTNet_Extraction(resnet, sample) for sample in x])
    dstnet.fit(feats, y)
    dstnet.save('model/dstnet.h5')
    
def test_script(x):
    resnet, dstnet = build_models(load_weights=True)
    feats = DSTNet_Extraction(x)
    feats = feats[np.newaxis,...]
    preds = np.argmax(DSTNet_Recognition(feats), axis=1)
    return preds
    
if __name__ == '__main__':
    train_script()
