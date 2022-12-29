import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from string_globals import *
import tensorflow as tf
import tensorflow_datasets as tfds
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
        a list of strings, each string is a style name.
    img_dir
        the directory where the images are stored.
    
    Returns
    -------
        A list of tuples, where each tuple contains the path to an image, the style of the image, and the
    name of the image.
    
    '''
    ret=[]
    all_styles_images=[img_dir+"/"+s for s in styles]
    for dir,style in zip(all_styles_images,styles):
        ret+=[(dir+"/"+image,style,image) for image in os.listdir(dir) if image.endswith(('.png','.jpg','.JPG','.PNG'))]
    return ret

def load_img(path_to_img, max_dim=256):
    '''It takes in a path to an image, loads the image, and then resizes it to a specified maximum
    dimension
    
    Parameters
    ----------
    path_to_img
        The path to the image you want to load.
    max_dim, optional
        The maximum dimension of the image. If the image is larger than this, it will be resized down.
    
    Returns
    -------
        The image is being returned.
    
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
    '''It takes a list of styles, finds all the images in those styles, loads them, resizes them, and saves
    them as numpy arrays
    
    Parameters
    ----------
    max_dim
        the maximum dimension of the image.
    styles
        a list of styles to process.
    root
        the directory where the npz files will be saved
    root_img_dir
        the directory where the images are stored
    
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
    '''It returns a list of paths to all the npz files in the given directory
    
    Parameters
    ----------
    max_dim, optional
        the maximum dimension of the images.
    styles
        the list of styles to use
    root
        the root directory of the npz files
    no_smote, optional
        If True, only returns paths to images that are not SMOTE samples.
    
    Returns
    -------
        A list of paths to the npz files.
    
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
    '''> It returns a list of tuples, where each tuple is a path to a numpy array and the style of the
    image
    
    Parameters
    ----------
    max_dim, optional
        the maximum dimension of the images.
    styles
        the list of styles to use
    root
        the root directory of the npz files
    no_smote, optional
        if True, then we don't include the SMOTE samples in the training set.
    
    Returns
    -------
        A list of tuples, where each tuple is a path to a numpy array and the style of the image.
    
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
    '''It takes a list of paths and a one-hot encoder and returns a generator that yields a tuple of
    (image, one-hot encoded style)
    
    Parameters
    ----------
    paths
        a list of tuples, where the first element is the path to the image, and the second element is the
    style of the image.
    ohencoder
        a OneHotEncoder object that will be used to convert the style labels to one-hot vectors
    
    Returns
    -------
        A generator function that returns a tuple of the image and the one hot encoded style.
    
    '''
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
    '''It returns a tf.data.Dataset object that can be used to load images from the npz files
    
    Parameters
    ----------
    max_dim, optional
        the maximum dimension of the images.
    styles
        a list of styles to include in the dataset.
    limit, optional
        the number of images to load
    root
        the directory where the npz files are stored
    no_smote, optional
        If True, then the dataset will not be SMOTE-balanced.
    
    Returns
    -------
        A tf.data.Dataset object.
    
    '''
    if root == 'deep_weeds':
        new_shape=[max_dim,max_dim]
        big_ds=[tf.image.resize(d['image'],new_shape)/255 for d in tfds.load('deep_weeds',split='all', shuffle_files=True) if d['label'] in styles][:limit]
        return tf.data.Dataset.from_tensor_slices(big_ds)
    paths=get_npz_paths(max_dim,styles,root,no_smote)
    shuffle(paths)
    paths=paths[:limit]
    gen=generator(paths)
    image_size=(max_dim,max_dim,3)
    return tf.data.Dataset.from_generator(gen,output_signature=(tf.TensorSpec(shape=image_size)))

def get_loader_labels(max_dim=64,styles=all_styles_faces_2,limit=100,root=faces_npz_dir_2,no_smote=True):
    '''It returns a tf.data.Dataset object that can be used to load images and labels from a directory of
    npz files
    
    Parameters
    ----------
    max_dim, optional
        the maximum dimension of the images.
    styles
        the list of styles to use
    limit, optional
        the number of images to load
    root
        the directory where the npz files are stored
    no_smote, optional
        if True, then the dataset will not be SMOTE-augmented.
    
    Returns
    -------
        A tf.data.Dataset object.
    
    '''
    ohencoder=OneHotEncoder(styles)
    if root == 'deep_weeds':
        new_shape=[max_dim,max_dim]
        big_ds=[(tf.image.resize(d['image'],new_shape)/255, ohencoder.transform(d['label'].numpy())) for d in tfds.load('deep_weeds',split='all', shuffle_files=True) if d['label'] in styles][:limit]
        def gen():
            for img,sty in big_ds:
                yield (img,sty)
    else:
        paths=get_npz_paths_labels(max_dim,styles,root,no_smote)
        shuffle(paths)
        paths=paths[:limit]
        gen=generator_labels(paths,ohencoder)
    image_size=(max_dim,max_dim,3)
    output_sig_shapes=tuple([tf.TensorSpec(shape=image_size),tf.TensorSpec(shape=(len(styles)))])
    return tf.data.Dataset.from_generator(gen,output_signature=output_sig_shapes)

def get_loader_oversample(max_dim,styles_quantity_dict,root): #this doesnt use smote it just gets extras/duplicates
    '''It takes a dictionary of styles and quantities, and returns a dataset of images of those styles,
    with the quantities specified
    
    Parameters
    ----------
    max_dim
        the maximum dimension of the image. The image will be resized to this dimension.
    styles_quantity_dict
        a dictionary of the style and the quantity of images you want for that style.
    root
        the root directory of the dataset
    
    Returns
    -------
        A dataset of images of the specified size and quantity.
    
    '''
    if root == 'deep_weeds':
        new_shape=[max_dim,max_dim]
        styles_to_lists={
            s: [tf.image.resize(d['image'],new_shape)/255 for d in tfds.load('deep_weeds',split='all', shuffle_files=True) if d['label'] == s] for s in styles_quantity_dict
        }
        big_ds=[]
        for style,quantity in styles_quantity_dict.items():
            count=0
            while count<quantity:
                add=len(styles_to_lists[style])
                if add+count>quantity:
                    add=quantity-count
                big_ds+=styles_to_lists[style][:add]
                count+=add
        return tf.data.Dataset.from_tensor_slices(big_ds)   
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
    '''It takes a dictionary of styles and quantities, and returns a dataset of images and labels that has
    the same number of images for each style
    
    Parameters
    ----------
    max_dim
        the maximum dimension of the image. The image will be resized to be square with this dimension.
    styles_quantity_dict
        a dictionary of the styles you want to use and how many images you want to use for each style.
    root
        the root directory of the dataset
    
    Returns
    -------
        A tf.data.Dataset object that can be iterated over to get the images and labels.
    
    '''
    ohencoder=OneHotEncoder([s for s in styles_quantity_dict.keys()])
    if root == 'deep_weeds':
        new_shape=[max_dim,max_dim]
        styles_to_lists={
            s: [(tf.image.resize(d['image'],new_shape)/255, ohencoder.transform(d['label'].numpy())) for d in tfds.load('deep_weeds',split='all', shuffle_files=True) if d['label'] == s] for s in styles_quantity_dict
        }
        big_ds=[]
        for style,quantity in styles_quantity_dict.items():
            count=0
            while count<quantity:
                add=len(styles_to_lists[style])
                if add+count>quantity:
                    add=quantity-count
                big_ds+=styles_to_lists[style][:add]
                count+=add
        def gen():
            for img,sty in big_ds:
                yield (img,sty)
    else:
        paths=[]
        for style,quantity in styles_quantity_dict.items():
            new_paths=get_npz_paths_labels(max_dim,[style],root)
            while len(new_paths) < quantity:
                new_paths+=get_npz_paths_labels(max_dim,[style],root)
            new_paths=new_paths[:quantity]
            paths+=new_paths
        shuffle(paths)
        gen=generator_labels(paths,ohencoder)
    image_size=(max_dim,max_dim,3)
    output_sig_shapes=tuple([tf.TensorSpec(shape=image_size),tf.TensorSpec(shape=(len(styles_quantity_dict)))])
    return tf.data.Dataset.from_generator(gen,output_signature=output_sig_shapes)