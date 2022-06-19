import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from string_globals import *
import tensorflow as tf
import sys
import numpy as np
import pandas as pd

def get_img_paths(styles):
    ret=[]
    all_styles_images=[img_dir+"/"+s for s in styles]
    for dir,style in zip(all_styles_images,styles):
        ret+=[(dir+"/"+image,style,image) for image in os.listdir(dir)]
    return ret

def load_img(path_to_img, max_dim=256):
    '''Loads the image from the given path, scales it to a square, and randomly crops a square of the given
    max dimension
    
    Parameters
    ----------
    path_to_img
        the path to the image you want to load.
    max_dim, optional
        The maximum dimension of the image.
    
    '''
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    #img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = min(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(1+(shape * scale), tf.int32)
    print(new_shape)

    img = tf.image.resize(img, new_shape,method='bilinear')
    img=tf.image.random_crop(img,size=(max_dim,max_dim,3))
    return img

def styles_to_npz(styles=all_styles):
    paths=get_img_paths(styles)
    for (src,style,jpg) in paths:
        name=jpg[:jpg.find(".")]
        img=load_img(src)
        np.save("{}/{}/{}".format(npz_root,style,name),img)

if __name__ == "__main__":
    styles=[s for s in set(sys.argv).intersection(set(all_styles))]
    styles_to_npz(styles)