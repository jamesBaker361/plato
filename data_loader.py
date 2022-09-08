import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from string_globals import *
import tensorflow as tf
import numpy as np
from random import Random
#from sklearn.preprocessing import OneHotEncoder

def shuffle(array):
    return Random(1234).shuffle(array)

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

def get_npz_paths(max_dim=64,styles=all_styles_faces_2,root=faces_npz_dir_2,no_smote=True):
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
    all_styles_npz_local=[root+"/"+s for s in styles]
    for dir,style in zip(all_styles_npz_local,styles):
        ret+=[dir+"/"+image for image in os.listdir(dir) if image.endswith(('{}.npy'.format(max_dim)))]
    if no_smote:
        okay=[]
        for r in ret:
            if r.find(smote_sample)==-1:
                okay.append(r)
        return okay
    return ret

def get_npz_paths_labels(max_dim=64,styles=all_styles_faces_2,root=faces_npz_dir_2,no_smote=True):
    '''> It takes a list of styles and a maximum dimension, and returns a list of tuples of the form
    (path,style) for all the images in the styles list
    
    Parameters
    ----------
    max_dim
        the maximum dimension of the image. The images are square, so this is the height and width.
    styles
        a list of the styles you want to train on.
    root
        the root directory of the npz files
    
    Returns
    -------
        A list of tuples, where each tuple is a path to an image and the style of that image.
    
    '''
    ret=[]
    all_styles_npz=[root+"/"+s for s in styles]
    for dir,style in zip(all_styles_npz,styles):
        ret+=[(dir+"/"+image,style )for image in os.listdir(dir) if image.endswith(('{}.npy'.format(max_dim)))]
    if no_smote:
        ret=[r for r in ret if r[0].find(smote_sample)==-1]
    return ret


class OneHotEncoder:
    def __init__(self,categories=None):
        if categories is not None:
            self.fit(categories)

    def fit(self,categories):
        encoding={name:[0.0 for _ in categories] for name in categories}
        for x,key in enumerate(encoding.keys()):
            encoding[key][x]=1.0
        self.encoding=encoding

    def transform(self,datapoint):
        return self.encoding[datapoint]

def generator_labels(paths,ohencoder):
    def _generator():
        for p,style in paths:
            yield (np.load(p)/255, ohencoder.transform(style))
    return _generator


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

def get_loader(max_dim=64,styles=all_styles_faces_2,limit=100,root=faces_npz_dir_2,no_smote=True):
    '''It returns a tensorflow dataset that generates images from the npz files in the specified directory
    
    Parameters
    ----------
    max_dim
        the maximum dimension of the image.
    styles
        a list of strings, each string is a style name.
    limit
        the number of images to load
    root
        the root directory of the dataset
    
    Returns
    -------
        A dataset of images of size (max_dim,max_dim,3)
    
    '''
    paths=get_npz_paths(max_dim,styles,root,no_smote)
    shuffle(paths)
    paths=paths[:limit]
    gen=generator(paths)
    image_size=(max_dim,max_dim,3)
    return tf.data.Dataset.from_generator(gen,output_signature=(tf.TensorSpec(shape=image_size)))

def get_loader_labels(max_dim=64,styles=all_styles_faces_2,limit=100,root=faces_npz_dir_2,no_smote=True):
    paths=get_npz_paths_labels(max_dim,styles,root,no_smote)
    shuffle(paths)
    paths=paths[:limit]
    ohencoder=OneHotEncoder(styles)
    gen=generator_labels(paths,ohencoder)
    image_size=(max_dim,max_dim,3)
    output_sig_shapes=tuple([tf.TensorSpec(shape=image_size),tf.TensorSpec(shape=(len(styles)))])
    return tf.data.Dataset.from_generator(gen,output_signature=output_sig_shapes)

def get_loader_oversample(max_dim,styles_quantity_dict,root): #this doesnt use smote it just gets extras/duplicates
    paths=[]
    for style,quantity in styles_quantity_dict.items():
        new_paths=get_npz_paths(max_dim,[style],root)
        while len(new_paths) < quantity:
            new_paths+=get_npz_paths(max_dim,[style],root)
        new_paths=new_paths[:quantity]
        paths+=new_paths
    shuffle(paths)
    gen=generator(paths)
    image_size=(max_dim,max_dim,3)
    return tf.data.Dataset.from_generator(gen,output_signature=(tf.TensorSpec(shape=image_size)))

def get_loader_oversample_labels(max_dim,styles_quantity_dict,root): #this doesnt use smote it just gets extras/duplicates
    paths=[]
    for style,quantity in styles_quantity_dict.items():
        new_paths=get_npz_paths_labels(max_dim,[style],root)
        while len(new_paths) < quantity:
            new_paths+=get_npz_paths_labels(max_dim,[style],root)
        new_paths=new_paths[:quantity]
        paths+=new_paths
    shuffle(paths)
    ohencoder=OneHotEncoder([s for s in styles_quantity_dict.keys()])
    gen=generator_labels(paths,ohencoder)
    image_size=(max_dim,max_dim,3)
    output_sig_shapes=tuple([tf.TensorSpec(shape=image_size),tf.TensorSpec(shape=(len(styles_quantity_dict)))])
    return tf.data.Dataset.from_generator(gen,output_signature=output_sig_shapes)