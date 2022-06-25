import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from string_globals import *
import tensorflow as tf
import sys
import numpy as np
import pandas as pd
from random import shuffle

def get_img_paths(styles):
    '''It takes a list of styles and returns a list of tuples of the form (image_path,style,image_name)
    
    Parameters
    ----------
    styles
        A list of the styles you want to use.
    
    Returns
    -------
        A list of tuples. Each tuple contains the path to the image, the style of the image, and the name
    of the image.
    
    '''
    ret=[]
    all_styles_images=[img_dir+"/"+s for s in styles]
    for dir,style in zip(all_styles_images,styles):
        ret+=[(dir+"/"+image,style,image) for image in os.listdir(dir) if image.endswith(('.png','.jpg','.JPG','.PNG'))]
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

def styles_to_npz(max_dim,styles=all_styles):
    '''> It takes a list of styles, gets the paths to all the images in those styles, and then saves the
    images as numpy arrays in a folder named after the style
    
    Parameters
    ----------
    styles
        a list of styles to convert.
    
    '''
    paths=get_img_paths(styles)
    for (src,style,jpg) in paths:
        name=jpg[:jpg.find(".")]
        new_np="{}/{}/{}.{}.npy".format(npz_root,style,name,max_dim)
        if os.path.exists(new_np):
            continue
        try:
            img=load_img(src,max_dim)
        except:
            print(name)
            continue
        np.save(new_np,img)

def get_npz_paths(max_dim,styles):
    ret=[]
    all_styles_npz=[npz_root+"/"+s for s in styles]
    for dir,style in zip(all_styles_npz,styles):
        ret+=[dir+"/"+image for image in os.listdir(dir) if image.endswith(('{}.npy'.format(max_dim)))]
    return ret

def generator(paths):
    def _generator():
        for p in paths:
            yield np.load(p)
    return _generator

def get_loader(max_dim,styles,limit,batch_size):
    paths=get_npz_paths(max_dim,styles)
    shuffle(paths)
    paths=paths[:limit]
    gen=generator(paths)
    return tf.data.Dataset.from_generator(gen,output_signature=(tf.TensorSpec(shape=image_size))).batch(batch_size,drop_remainder=True).shuffle(10,reshuffle_each_iteration=False)

    




if __name__ == "__main__":
    styles=[s for s in set(sys.argv).intersection(set(all_styles))]
    for m in [64,128,256]:
        styles_to_npz(m,styles)