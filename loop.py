import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from cvae import CVAE
from data_loader import *
import time
import datetime
import random

import argparse

parser = argparse.ArgumentParser(description='get some args')

parser.add_argument("--name",type=str,default="name")

for names,default in zip(["batch_size","max_dim","epochs","latent_dim","quantity"],[16,64,10,2,100]):
    parser.add_argument("--{}".format(names),type=int,default=default)

args = parser.parse_args()



num_examples_to_generate = 16

optimizer = tf.keras.optimizers.Adam(0.00001,clipnorm=1.0)

image_size=(args.max_dim,args.max_dim,3)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
args.name+=current_time

os.makedirs(gen_img_dir+"/"+args.name)

train_log_dir = 'logs/gradient_tape/' + args.name + '/train'
test_log_dir = 'logs/gradient_tape/' + args.name + '/test'


logpx_z_log_dir='logs/gradient_tape/' + args.name + '/logpx_z'
logpz_log_dir='logs/gradient_tape/' + args.name + '/logpz'
logqz_x_log_dir='logs/gradient_tape/' + args.name + '/logqz_x'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

scalar_names=["logpx_z","logpz","logqz_x","kl","reconst"]

dirs={}
writers={}
metrics={}
for name in scalar_names:
    dirs[name]='logs/gradient_tape/' + args.name + '/'+name
    writers[name]=tf.summary.create_file_writer(dirs[name])
    metrics[name]=tf.keras.metrics.Mean(name,dtype=tf.float32)


train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)


def compute_loss(model, x,test=False):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    cross_ent=(x-x_logit)**2
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    #return tf.random.uniform(shape=[], minval=5, maxval=10, dtype=tf.float32)
    if test:
        for name,variable in zip(["logpx_z","logpz","logqz_x"],[-logpx_z,-logpz,logqz_x]):
            #with writers[name].as_default():
            metrics[name](tf.reduce_mean(variable))
            #tf.summary.scalar('loss', metrics[name].result(), step=epoch)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_loss_2(model,x,test=False):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    """def kl_loss(mean, log_var):
    kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
    return kl_loss
    """
    cross_ent=(x-x_logit)**2
    reconst = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    kl=-0.5 * tf.reduce_sum(1+logvar - mean**2 - tf.exp(logvar),axis=1)

    if test:
        for name,variable in zip(["reconst","kl"],[reconst,kl]):
            metrics[name](tf.reduce_mean(variable))

    return tf.reduce_mean(reconst+kl)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss_2(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, args.latent_dim])
model = CVAE(args.latent_dim,args.max_dim)

def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('{}/{}/image_at_epoch_{:04d}.png'.format(gen_img_dir,args.name,epoch))
    plt.show()

loader=get_loader(args.max_dim,["0"],args.quantity,mnist_npz_root)

test_dataset = loader.enumerate().filter(lambda x,y: x % 4 == 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(args.batch_size,drop_remainder=True)

train_dataset = loader.enumerate().filter(lambda x,y: x % 4 != 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(args.batch_size,drop_remainder=True)

# Pick a sample of the test set for generating output images
assert args.batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    break

generate_and_save_images(model, 0, test_sample)

for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)



    for test_x in test_dataset:
        test_loss(compute_loss_2(model, test_x,True))
    elbo = -test_loss.result()

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)

    for name in ["logpx_z","logpz","logqz_x","kl","reconst"]:
        with writers[name].as_default():
            tf.summary.scalar('loss', metrics[name].result(), step=epoch)
            metrics[name].reset_states()

    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)

    train_loss.reset_states()
    test_loss.reset_states()