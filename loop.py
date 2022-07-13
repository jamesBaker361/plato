import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from cvae import CVAE
from data_loader import *
import time
import datetime
import random
from vgg import *

import argparse

parser = argparse.ArgumentParser(description='get some args')

parser.add_argument("--name",type=str,default="name")
parser.add_argument("--dataset",type=str,default="mnist",help="name of dataset (mnist or art or faces)")
parser.add_argument("--genres",nargs='+',type=str,default=["0"],help="which digits/artistic genres ")
parser.add_argument("--diversity",type=bool,default=False,help="whether to use unconditional diversity loss")
parser.add_argument("--lambd",type=float,default=0.1,help="coefficient on diversity term")
parser.add_argument("--vgg_style",type=bool,default=False,help="whether to use vgg style reconstruction loss too")
parser.add_argument("--blocks",nargs='+',type=str,default=["block1_conv1"],help="blocks for vgg extractor")
parser.add_argument("--vgg_lambda",type=float,default=0.1,help="coefficient on vgg style loss")
parser.add_argument("--logdir",type=str,default='logs/gradient_tape/')

for names,default in zip(["batch_size","max_dim","epochs","latent_dim","quantity","diversity_batches","test_split"],[16,64,10,2,100,4,4]):
    parser.add_argument("--{}".format(names),type=int,default=default)

args = parser.parse_args()



num_examples_to_generate = 16

image_size=(args.max_dim,args.max_dim,3)

current_time = datetime.datetime.now().strftime("%H%M%S")
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
    global_batch_size=len(logical_gpus)*args.batch_size
else:
    global_batch_size=args.batch_size



train_log_dir = args.logdir + args.name + '/train'
test_log_dir = args.logdir + args.name + '/test'
diversity_log_dir=args.logdir + args.name + '/diversity'
vgg_log_dir=args.logdir+args.name+"/vgg"


logpx_z_log_dir=args.logdir + args.name + '/logpx_z'
logpz_log_dir=args.logdir + args.name + '/logpz'
logqz_x_log_dir=args.logdir + args.name + '/logqz_x'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
diversity_summary_writer=tf.summary.create_file_writer(diversity_log_dir)
vgg_summary_writer=tf.summary.create_file_writer(vgg_log_dir)



#scalar_names=["logpx_z","logpz","logqz_x","kl","reconst"]

"""dirs={}
writers={}
metrics={}
for name in scalar_names:
    dirs[name]=args.logdir + args.name + '/'+name
    writers[name]=tf.summary.create_file_writer(dirs[name])
    metrics[name]=tf.keras.metrics.Mean(name,dtype=tf.float32)"""





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
    if args.vgg_style:
        style_extractor=vgg_layers(args.blocks,image_size)

    '''for name in scalar_names:
        dirs[name]=args.logdir + args.name + '/'+name
        writers[name]=tf.summary.create_file_writer(dirs[name])
        metrics[name]=tf.keras.metrics.Mean(name,dtype=tf.float32)'''


    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    diversity_loss=tf.keras.metrics.Mean("diversity_loss",dtype=tf.float32)
    vgg_loss=tf.keras.metrics.Mean("vgg_loss",dtype=tf.float32)

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

        """if test:
            for name,variable in zip(["reconst","kl"],[reconst,kl]):
                metrics[name](tf.reduce_mean(variable))"""

        per_example_loss= reconst+kl

        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    def compute_diversity_loss(samples,z):
        '''based off of https://arxiv.org/abs/1901.09024
        '''
        try:
            batch_size=len(samples)
        except TypeError:
            batch_size=samples.shape[0]
        _loss=tf.constant(0.0,dtype=tf.float32) #[-1.0* tf.reduce_mean(tf.square(tf.subtract(samples, samples)))]
        for i in range(batch_size):
            for j in range(i+1,batch_size):
                _loss+=tf.norm(samples[i]-samples[j])/tf.norm(z[i]-z[j])
        loss=[-_loss*args.lambd]
        return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

    def compute_vgg_style_loss(x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        x_style=style_extractor(x)
        x_logit_style=style_extractor(x_logit)
        cross_ent=(x_style-x_logit_style)**2
        reconst=tf.reduce_sum(args.vgg_lambda * cross_ent,axis=[1,2,3])
        return tf.nn.compute_average_loss(reconst,global_batch_size=global_batch_size)
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

def diversity_step(z):
    with tf.GradientTape() as tape:
        samples=model.sample(z)
        loss=compute_diversity_loss(samples,z)
    gradients = tape.gradient(loss, model.decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.decoder.trainable_variables))
    diversity_loss(loss)
    return loss

def vgg_style_step(x):
    with tf.GradientTape() as tape:
        loss=compute_vgg_style_loss(x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    vgg_loss(loss)
    return loss

@tf.function
def distributed_diversity_step(z):
    per_replica_losses = strategy.run(diversity_step, args=(z,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

@tf.function
def distributed_train_step(x):
  per_replica_losses = strategy.run(train_step, args=(x,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

@tf.function
def distributed_test_step(x):
  return strategy.run(test_step, args=(x,))

@tf.function
def distributed_vgg_style_step(x):
    per_replica_losses = strategy.run(vgg_style_step, args=(x,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

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
    "art":npz_root,
    "faces":faces_npz_dir
}

loader=get_loader(args.max_dim,args.genres,args.quantity,root_dict[args.dataset])

test_dataset = loader.enumerate().filter(lambda x,y: x % args.test_split == 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

train_dataset = loader.enumerate().filter(lambda x,y: x % args.test_split != 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

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
        if args.vgg_style:
            distributed_vgg_style_step(train_x)
    end_time = time.time()
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
    if args.vgg_style:
        with vgg_summary_writer.as_default():
            tf.summary.scalar('loss',vgg_loss.result(),step=epoch)

    if args.diversity:
        for _ in range(args.diversity_batches):
            z=tf.random.normal(shape=[8, args.latent_dim])
            distributed_diversity_step(z)

        with diversity_summary_writer.as_default():
            tf.summary.scalar('loss',diversity_loss.result(),step=epoch)

    for test_x in test_dataset:
        distributed_test_step(test_x)
    elbo = -test_loss.result()

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)

    '''for name in ["logpx_z","logpz","logqz_x","kl","reconst"][3:]:
        with writers[name].as_default():
            tf.summary.scalar('loss', metrics[name].result(), step=epoch)
            metrics[name].reset_states()'''

    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)
    generate_from_noise(model,epoch,random_vector_for_generation)

    train_loss.reset_states()
    test_loss.reset_states()
    diversity_loss.reset_states()
    vgg_loss.reset_states()