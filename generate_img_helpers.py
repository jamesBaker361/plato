import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from classifier import get_discriminator
from cvae import CVAE
from data_loader import *
import time
from random import randrange
import datetime
from c3vae import C3VAE
from attvae import *
from fid import calculate_fid
from vgg import *
from optimizer_config import get_optimizer
import logging
from smote_data import *
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

def get_generate_and_save_images(print_debug,args):
    def generate_and_save_images(model, epoch, test_sample,apply_sigmoid):
        '''It takes a batch of images, encodes them, samples from the latent space, and then decodes them
        
        Parameters
        ----------
        model
            The model that we are training.
        epoch
            The current epoch number.
        test_sample
            a batch of test images
        apply_sigmoid
            If True, the output of the generator will go through a sigmoid function. 
        
        '''
        predictions = model(test_sample)
        print_debug("real prediction shape",predictions.shape)
        batch_size=predictions.shape[0]
        fig = plt.figure(figsize=(4, batch_size//4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, batch_size//4, i + 1)
            plt.imshow(predictions[i])
            plt.axis('off')

        plt.savefig('{}/{}/image_at_epoch_{:04d}.png'.format(gen_img_dir,args.name,epoch))
        plt.show()
    return generate_and_save_images

def get_generate_from_noise(args,num_examples_to_generate):
    def generate_from_noise(model,epoch,random_vector,apply_sigmoid):
        '''It takes a model, an epoch number, a random vector, and a boolean flag, and then generates a grid of
        images from the random vector, and saves the grid to a file.
        
        Parameters
        ----------
        model
            the model we're using to generate images
        epoch
            the current epoch number
        random_vector
            A vector of shape (1,latent_dim) that will be used to generate an image.
        apply_sigmoid
            If True, the output of the generator will be passed through a sigmoid function. This is useful if
        the output of the generator is a probability.
        
        '''
        if args.c3vae:
            indices=[randrange(len(args.genres)) for _ in range(num_examples_to_generate)]
            random_classes=tf.one_hot(indices,len(args.genres))
            predictions=model.sample(random_vector,apply_sigmoid, random_classes)
        else:
            predictions = model.sample(random_vector,apply_sigmoid)

        batch_size=predictions.shape[0]
        fig = plt.figure(figsize=(4, batch_size//4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, batch_size//4, i + 1)
            plt.imshow(predictions[i])
            plt.axis('off')

        plt.savefig('{}/{}/gen_image_at_epoch_{:04d}.png'.format(gen_img_dir,args.name,epoch))
        plt.show()
    return generate_from_noise