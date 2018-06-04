# -*- coding: utf-8 -*-
import os
import pickle

def save_dict(file_path, mdict):
    with open(file_path, 'wb') as f:
        pickle.dump(mdict, f)
        
def load_dict(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def load_dict_txt(file_path):
    mdict = {}
    f = open(file_path, 'r')
    for line in f:
        key, value = line.split(' ')
        mdict[key] = int(value)
    return mdict
