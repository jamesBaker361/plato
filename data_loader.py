import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from string_globals import *
import tensorflow as tf
import sys
import numpy as np
import pandas as pd
from random import shuffle

def get_img_paths(styles,img_dir=img_dir):
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
    if len(img.shape)==4:
        img=img[0]
    #img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize_with_pad(img, max_dim,max_dim)
    #img=tf.image.random_crop(img,size=(max_dim,max_dim,3))
    return img

def styles_to_npz(max_dim,styles=all_styles,root=npz_root,root_img_dir=img_dir):
    '''> It takes a list of styles, gets the paths to all the images in those styles, and then saves the
    images as numpy arrays in a folder named after the style
    
    Parameters
    ----------
    styles
        a list of styles to convert.
    
    '''
    paths=get_img_paths(styles,root_img_dir)
    for (src,style,jpg) in paths:
        name=jpg[:jpg.rfind(".")]
        new_np="{}/{}/{}.{}.npy".format(root,style,name,max_dim)
        if os.path.exists(new_np):
            continue
        try:
            img=load_img(src,max_dim)
        except:
            print("could not load",name)
            continue
        np.save(new_np,img)
        print("saved",new_np)

def get_npz_paths(max_dim,styles,root=npz_root):
    '''It takes a list of styles and a maximum dimension, and returns a list of paths to the npz files for
    those styles
    
    Parameters
    ----------
    max_dim
        the maximum dimension of the image. The images are square, so this is the length of the longest
    side.
    styles
        a list of the styles you want to use.
    root
        the root directory of the npz files
    
    Returns
    -------
        A list of paths to the numpy files.
    
    '''
    ret=[]
    all_styles_npz=[root+"/"+s for s in styles]
    for dir,style in zip(all_styles_npz,styles):
        ret+=[dir+"/"+image for image in os.listdir(dir) if image.endswith(('{}.npy'.format(max_dim)))]
    return ret

def generator(paths):
    '''It takes a list of paths to numpy arrays, and returns a function that loads those arrays and divides
    them by 255
    
    Parameters
    ----------
    paths
        a list of paths to the numpy arrays
    
    Returns
    -------
        A function that takes no arguments and returns a generator that yields the normalized numpy arrays.
    
    '''
    def _generator():
        for p in paths:
            yield np.load(p) /255
    return _generator

def get_loader(max_dim,styles,limit,root):
    paths=get_npz_paths(max_dim,styles,root)
    shuffle(paths)
    paths=paths[:limit]
    gen=generator(paths)
    image_size=(max_dim,max_dim,3)
    return tf.data.Dataset.from_generator(gen,output_signature=(tf.TensorSpec(shape=image_size)))

    




if __name__ == "__main__":
    #styles=[s for s in set(sys.argv).intersection(set(all_styles))]
    #styles=all_styles_faces
    for m in [64,128,256]:
        styles_to_npz(m,all_styles_faces,faces_npz_dir,faces_dir)
        #styles_to_npz(m,styles)
        #styles_to_npz(m,all_digits,mnist_npz_root)