# -*- coding: utf-8 -*-
import os
import cv2
import h5py
import numpy as np
from utils import save_dict, load_dict, load_dict_txt
from tensorflow.contrib.keras.api.keras.preprocessing import image
"""
    Extract pictures from videos.
    Arguements:
        video_dir:  video directory
        save_dir:   images saved directory
        frame_gap:  each frame_gap extract a image in video
        frame_size: image saved size
        frames_per_group: time step for BLSTM
"""
def prepare_data(video_dir, save_dir='datas', frame_gap=10, frame_size=224, frames_per_group=16):
    # "video-group" dictionary
    frame_info = dict()
    # extract frame from directory
    for lidx, label in enumerate(os.listdir(video_dir)):
        # check data director, if not exist, create
        save_class_dir = os.path.join(save_dir, label)
        if not os.path.exists(save_class_dir):
            os.makedirs(save_class_dir)
            print('create dirctory: {}'.format(save_class_dir))
        # read from each video
        for vidx, video in enumerate(os.listdir(os.path.join(video_dir, label))):
            # read video
            cap = cv2.VideoCapture(os.path.join(video_dir, label, video))
            frame_count = 0
            while(cap.isOpened()):
                _, frame = cap.read()
                # image save each "frame_gap" time
                if frame is not None:
                    frame_count += 1
                    if(frame_count % frame_gap == 1):
                        group = frame_count // (frame_gap * frames_per_group)
                        index = (frame_count // frame_gap ) % frames_per_group
                        im = cv2.resize(frame, (frame_size,frame_size))
                        im = im[:,56:168,:]
                        im = cv2.resize(im, (frame_size,frame_size))
                        cv2.imwrite(os.path.join(save_class_dir, 'video{}_group{}_index{}.jpg'.format(vidx, group, index)), im)
                if (cv2.waitKey(1) & 0xFF == ord('q')) or frame is None:
                    break
            # When everything done, release the capture
            cap.release()
            print('process video: {}, frames: {}, frame_gap: {}, groups: {}'.format(os.path.join(video_dir, label, video), frame_count, frame_gap, group))
            frame_info[os.path.join(save_class_dir, 'video{}@{}'.format(vidx, lidx))] = frame_count // (frame_gap * frames_per_group)
    # dave dictionary
    save_dict('model/frame_info_{}.pkl'.format(frames_per_group), frame_info)
     

def data_loader(image_dir, image_size=224, time_step=16, class_nums=4, load_from_hdf5=False):
    def generate_example(path, group):
        img_path = ['{}_group{}_index{}.jpg'.format(path, group, idx) for idx in range(time_step)]
        imgs = [image.img_to_array(image.load_img(x)) for x in img_path]
        return imgs
    
    # load data from hdf5 file
    if load_from_hdf5:
        f = h5py.File('data.h5', 'r')
        x = f['data'][:]
        y = f['label'][:]
        f.close()
    else:
        frame_info = load_dict('model/frame_info.pkl')
        x, y = [], []
        for key, value in frame_info.items():
            path, label = key.split('@')
            label = int(label)
            if label >= class_nums or value==0:
                continue
            x += [generate_example(path, g) for g in range(value)]
            onehot_label = [0] * class_nums
            onehot_label[label] = 1
            y += [onehot_label]*value
        x = np.array(x, dtype=np.uint8)
        y = np.array(y, dtype=np.uint8)
        f = h5py.File('data.h5','w')
        f['data'] = x
        f['label'] = y
        f.close()
    return x, y

#prepare_data(video_dir='/home/zyl/datasets/kth', frame_gap=3, frames_per_group=12, save_dir='data_12')
#data_loader('data/', time_step=12)
