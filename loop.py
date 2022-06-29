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
parser.add_argument("--dataset",type=str,default="mnist",help="name of dataset (mnist or art)")
parser.add_argument("--genres",nargs='+',type=str,default=["0"],help="which digits/artistic genres ")

for names,default in zip(["batch_size","max_dim","epochs","latent_dim","quantity"],[16,64,10,2,100]):
    parser.add_argument("--{}".format(names),type=int,default=default)

args = parser.parse_args()



num_examples_to_generate = 16

image_size=(args.max_dim,args.max_dim,3)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
args.name+=current_time

os.makedirs(gen_img_dir+"/"+args.name)

physical_devices=tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device,True)
    except  RuntimeError as e:
        print(e)

logical_gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy()

if len(logical_gpus)>0:
    global_batch_size=strategy.num_replicas_in_sync()*args.batch_size
else:
    global_batch_size=args.batch_size

train_log_dir = 'logs/gradient_tape/' + args.name + '/train'
test_log_dir = 'logs/gradient_tape/' + args.name + '/test'


logpx_z_log_dir='logs/gradient_tape/' + args.name + '/logpx_z'
logpz_log_dir='logs/gradient_tape/' + args.name + '/logpz'
logqz_x_log_dir='logs/gradient_tape/' + args.name + '/logqz_x'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

scalar_names=["logpx_z","logpz","logqz_x","kl","reconst"][3:]

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







# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, args.latent_dim])

with strategy.scope():
    model = CVAE(args.latent_dim,args.max_dim)
    optimizer = tf.keras.optimizers.Adam(0.00001,clipnorm=1.0)

    for name in scalar_names:
        dirs[name]='logs/gradient_tape/' + args.name + '/'+name
        writers[name]=tf.summary.create_file_writer(dirs[name])
        metrics[name]=tf.keras.metrics.Mean(name,dtype=tf.float32)


    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    def compute_loss(x,test=False):
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

        per_example_loss= reconst+kl

        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
#end strategy scope

def train_step(x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return loss

def test_step(x):
    loss=compute_loss(x)
    test_loss(loss)
    return loss

@tf.function
def distributed_train_step(x):
  per_replica_losses = strategy.run(train_step, args=(x,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

@tf.function
def distributed_test_step(x):
  return strategy.run(test_step, args=(x,))

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

def generate_from_noise(model,epoch,random_vector):
    predictions = model.sample(random_vector)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('{}/{}/gen_image_at_epoch_{:04d}.png'.format(gen_img_dir,args.name,epoch))
    plt.show()

root_dict={
    "mnist":mnist_npz_root,
    "art":npz_root
}

loader=get_loader(args.max_dim,args.genres,args.quantity,root_dict[args.dataset])

test_dataset = loader.enumerate().filter(lambda x,y: x % 4 == 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(args.batch_size,drop_remainder=True)

train_dataset = loader.enumerate().filter(lambda x,y: x % 4 != 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(args.batch_size,drop_remainder=True)

# Pick a sample of the test set for generating output images
assert args.batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    break

train_dataset=strategy.experimental_distribute_dataset(train_dataset)
test_dataset=strategy.experimental_distribute_dataset(test_dataset)

generate_and_save_images(model, 0, test_sample)
generate_from_noise(model,0,random_vector_for_generation)

for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        distributed_train_step(train_x)
    end_time = time.time()
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)



    for test_x in test_dataset:
        distributed_test_step(test_x)
    elbo = -test_loss.result()

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)

    for name in ["logpx_z","logpz","logqz_x","kl","reconst"][3:]:
        with writers[name].as_default():
            tf.summary.scalar('loss', metrics[name].result(), step=epoch)
            metrics[name].reset_states()

    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)
    generate_from_noise(model,epoch,random_vector_for_generation)

    train_loss.reset_states()
    test_loss.reset_states()