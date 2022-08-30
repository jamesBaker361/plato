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
import datetime
import random
from fid import calculate_fid
from vgg import *
from optimizer_config import get_optimizer
import logging
from smote_data import *
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

import argparse

parser = argparse.ArgumentParser(description='get some args')

parser.add_argument("--name",type=str,default="name")
parser.add_argument("--save",type=str,default=False,help="whether to save this model")
parser.add_argument("--dataset",type=str,default="faces2",help="name of dataset (mnist or art or faces or faces2)")
parser.add_argument("--use_smote",type=bool,default=True,)
parser.add_argument("--genres",nargs='+',type=str,default=[],help="which digits/artistic genres ")

parser.add_argument("--diversity",type=bool,default=False,help="whether to use unconditional diversity loss")
parser.add_argument("--lambd",type=float,default=0.1,help="coefficient on diversity term")

parser.add_argument("--kl_weight",type=float,default=1.0,help="weight on kl term; if 0, defaults to dataset size/batch size")

parser.add_argument("--vgg_style",type=bool,default=False,help="whether to use vgg style reconstruction loss too")
parser.add_argument("--blocks",nargs='+',type=str,default=["block1_conv1"],help="blocks for vgg extractor")
parser.add_argument("--vgg_lambda",type=float,default=0.1,help="coefficient on vgg style loss")

parser.add_argument("--resnet",type=bool,default=False,help="whether to use resnet reconstruction loss too")
parser.add_argument("--resnet_blocks",type=str,nargs='+',default=["conv2_block1_out"],help="which blocks to use for resnet extractor")
parser.add_argument("--resnet_lambda",type=float,default=0.1,help="coefficient on resnet style loss")

parser.add_argument("--logdir",type=str,default='logs/gradient_tape/')

parser.add_argument("--opt_type",type=str,default="vanilla",help="whether to use normal lr (vanilla) decay cyclical or superconvergence")
parser.add_argument("--init_lr",type=float,default=0.00001,help="init lr for cyclical learning rate, decaying learning rate")
parser.add_argument("--max_lr",type=float,default=0.01,help="max lr for cyclical learning rate")
parser.add_argument("--min_mom",type=float,default=0.85, help="minimum momentum")
parser.add_argument("--max_mom",type=float,default=0.95,help="maximum momentum")
parser.add_argument("--phase_one_pct",type=float,default=0.4,help="what percent of cycle shold be the first phase for super convergence")
parser.add_argument("--decay_rate",type=float,default=0.9,help="decay rate for decaying LR")
parser.add_argument("--cycle_steps",type=int,default=1000,help="how many steps before decaying LR, or one cycle of LR decay and reverse decay")
parser.add_argument("--opt_name",type=str,default='adam',help="which optimizer to use, adam or sgd or rms")
parser.add_argument("--clipnorm",type=float,default=1.0,help="max gradient norm")

parser.add_argument("--fid",type=bool,default=False,help="whether to do FID scoring")
parser.add_argument("--fid_interval",type=int, default=10,help="FID scoring every X intervals")
parser.add_argument("--fid_sample_size",type=int,default=1000,help="how many images to do each FID scoring")

parser.add_argument("--apply_sigmoid",type=bool,default=False,help="whether to apply sigmoid when sampling")

parser.add_argument("--gan",type=bool,default=False,help="whether to implement GAN training or not")
parser.add_argument("--gan_start",type=int,default=150,help="# of epochs after which to start GAN training")
parser.add_argument("--lambda_gp",type=float, default=10.0,help="lambda on gradient penalty")
parser.add_argument("--n_critic",type=int, default=5,help="train the discriminator n times more than generator")
parser.add_argument("--level",type=str,default="dc",help="which architecture to use for disciriminator")
parser.add_argument("--weights",type=str,default=None,help="weights= imagenet or None for vgg discirimnaotor")
parser.add_argument("--extra_epochs",type=int,default=100,help="whether to train the gan for any epochs after training the vae")

parser.add_argument("--generate_smote",type=bool,default=False,help="whether to generate any synthetic images for smote")
parser.add_argument("--smote_minimum",type=int,default=100,help="bare minimum amount of examples in a class to generate smotes")
parser.add_argument("--smote_maximum",type=int,default=2500,help="amount of samples we want for each class")

parser.add_argument("--creativity",type=bool,default=False,help="whether to use elgammal creatvity loss")
parser.add_argument("--creativity_lambda",type=float,default=100,help="coefficient on creativty loss")
parser.add_argument("--classifier_path",type=str,default="../../../../../scratch/jlb638/plato/checkpoints/creativity_3_64",help="path to load classifier from")
parser.add_argument("--creativity_start",type=int,default=0,help="epoch when to start applying creativity loss")

parser.add_argument("--evaluation_imgs",type=int,default=0,help="how many images to generate at the end for evaluation")
parser.add_argument("--evaluation_path",type=str,default="./evaluation/",help="where to save evaluation images")

for names,default in zip(["batch_size","max_dim","epochs","latent_dim","quantity","diversity_batches","test_split"],[16,64,10,2,250,4,8]):
    parser.add_argument("--{}".format(names),type=int,default=default)

args = parser.parse_args()

if len(args.genres)==0:
    args.genres=dataset_default_all_styles[args.dataset]

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

actual_length=len(get_npz_paths(args.max_dim,args.genres, root_dict[args.dataset]))
actual_length=min(actual_length,args.quantity)
print("actual length of dataset= {}".format(actual_length))

if args.kl_weight==0.0:
    args.kl_weight=actual_length/global_batch_size

train_log_dir = args.logdir + args.name + '/train'
test_log_dir = args.logdir + args.name + '/test'
diversity_log_dir=args.logdir + args.name + '/diversity'
vgg_log_dir=args.logdir+args.name+"/vgg"
resnet_log_dir=args.logdir+args.name+"/resnet"
creativity_log_dir=args.logdir+args.name+"/creativity"



logpx_z_log_dir=args.logdir + args.name + '/logpx_z'
logpz_log_dir=args.logdir + args.name + '/logpz'
logqz_x_log_dir=args.logdir + args.name + '/logqz_x'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
diversity_summary_writer=tf.summary.create_file_writer(diversity_log_dir)
vgg_summary_writer=tf.summary.create_file_writer(vgg_log_dir)
resnet_summary_writer=tf.summary.create_file_writer(resnet_log_dir)
creativity_summary_writer=tf.summary.create_file_writer(creativity_log_dir)

if args.fid:
    fid_log_dir=args.logdir+args.name+"/fid"
    fid_summary_writer=tf.summary.create_file_writer(fid_log_dir)
    random_vector_fid=tf.random.normal(shape=[args.fid_sample_size, args.latent_dim])

if args.gan:
    gp_log_dir=args.logdir+args.name+"/gp"
    disc_log_dir=args.logdir+args.name+"/discriminator_loss"
    gen_log_dir=args.logdir+args.name+"/generator_loss"

    gp_summary_writer=tf.summary.create_file_writer(gp_log_dir)
    disc_summary_writer=tf.summary.create_file_writer(disc_log_dir)
    gen_summary_writer=tf.summary.create_file_writer(gen_log_dir)


# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, args.latent_dim])



with strategy.scope():
    model = CVAE(args.latent_dim,args.max_dim)
    optimizer=get_optimizer(
        args.opt_name,
        args.opt_type,
        args.init_lr,
        args.max_lr,
        args.min_mom,
        args.max_mom,
        args.decay_rate,
        args.cycle_steps,
        args.phase_one_pct,
        args.clipnorm)
    if args.vgg_style:
        vgg_style_extractor=vgg_layers(args.blocks,image_size)

    if args.resnet:
        resnet_style_extractor=resnet_layers(args.resnet_blocks,image_size)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    diversity_loss=tf.keras.metrics.Mean("diversity_loss",dtype=tf.float32)
    vgg_loss=tf.keras.metrics.Mean("vgg_loss",dtype=tf.float32)
    resnet_loss=tf.keras.metrics.Mean("resnet_loss",dtype=tf.float32)
    fid_loss=tf.keras.metrics.Mean("fid_loss",dtype=tf.float32)
    gp_loss=tf.keras.metrics.Mean("gp_loss",dtype=tf.float32)
    disc_loss=tf.keras.metrics.Mean("disc_loss",dtype=tf.float32)
    gen_loss=tf.keras.metrics.Mean("gen_loss",dtype=tf.float32)
    creativity_loss=tf.keras.metrics.Mean("creativity_loss",dtype=tf.float32)

    def compute_loss(x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z,args.apply_sigmoid)
        cross_ent=(x-x_logit)**2
        reconst = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        kl= -0.5* args.kl_weight * tf.reduce_sum(1+logvar - mean**2 - tf.exp(logvar),axis=1)

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

    def compute_style_loss(x,extractor,extractor_lambda):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z,args.apply_sigmoid)
        x_style=extractor(x)
        x_logit_style=extractor(x_logit)
        cross_ent=(x_style-x_logit_style)**2
        reconst=tf.reduce_sum(extractor_lambda * cross_ent,axis=[1,2,3])
        return tf.nn.compute_average_loss(reconst,global_batch_size=global_batch_size)

    def compute_vgg_style_loss(x):
        return compute_style_loss(x,vgg_style_extractor,args.vgg_lambda)

    def compute_resnet_style_loss(x):
        return compute_style_loss(x,resnet_style_extractor,args.resnet_lambda)

    if args.gan:
        disc=get_discriminator(args.level,(args.max_dim,args.max_dim,3),args.weights)
        disc_optimizer=get_optimizer(
        args.opt_name,
        args.opt_type,
        args.init_lr,
        args.max_lr,
        args.min_mom,
        args.max_mom,
        args.decay_rate,
        args.cycle_steps,
        args.phase_one_pct,
        args.clipnorm)
        def compute_gradient_penalty(real_x,fake_x):
            alpha=tf.random.uniform((args.batch_size,1,1,1),minval=0.0,maxval=1.0)
            differences=fake_x-real_x
            interpolates=real_x+(alpha * differences)
            gradients=tf.gradients(disc(interpolates),interpolates)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            gradient_penalty = [tf.reduce_mean((slopes-1.)**2)]
            return tf.nn.compute_average_loss(gradient_penalty,global_batch_size=global_batch_size)

        def compute_gan_loss(labels):
            loss=[tf.reduce_mean(labels)]
            return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

    if args.creativity:
        classifier=tf.keras.models.load_model(args.classifier_path)
        print('loaded from ',args.classifier_path)

        def compute_creativity_loss(class_labels):
            uniform=tf.fill([args.batch_size,len(args.genres)],1.0/len(args.genres))
            cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            loss=cce(class_labels,uniform)
            return tf.nn.compute_average_loss(args.creativity_lambda * loss, global_batch_size=global_batch_size)



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
        samples=model.sample(z,args.apply_sigmoid)
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

def resnet_style_step(x):
    with tf.GradientTape() as tape:
        loss=compute_resnet_style_loss(x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    resnet_loss(loss)
    return loss

def fid_step(model,noise,loader):
    images1=model.sample(noise,False)
    for i in loader.shuffle(1000,reshuffle_each_iteration=True).batch(args.fid_sample_size):
        images2=i
        break
    loss=calculate_fid(images1,images2,(args.max_dim,args.max_dim,3))
    return loss

if args.gan:
    def adversarial_step(x,gen_training=False):
        z=tf.random.truncated_normal((args.batch_size,args.latent_dim))
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            fake_x=model.sample(z,args.apply_sigmoid)

            fake_labels=disc(fake_x)
            real_labels=disc(x)

            fake_d_loss=compute_gan_loss(fake_labels)
            real_d_loss=compute_gan_loss(real_labels)
            gp=compute_gradient_penalty(x,fake_x)
            g_loss=-fake_d_loss
            d_loss=fake_d_loss-real_d_loss + (args.lambda_gp*gp)
        gen_loss(g_loss)
        disc_loss(d_loss)
        gp_loss(gp)
        disc_gradients=disc_tape.gradient(d_loss,disc.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_gradients,disc.trainable_variables))
        if gen_training:
            gen_gradients=gen_tape.gradient(g_loss,model.decoder.trainable_variables)
            optimizer.apply_gradients(zip(gen_gradients, model.decoder.trainable_variables))
        return g_loss,d_loss
    
    @tf.function
    def distributed_adversarial_step(x,gen_training):
        per_replica_losses = strategy.run(adversarial_step, args=(x,gen_training,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                            axis=None)

if args.creativity:
    def creatvity_step(x):
        with tf.GradientTape(persistent=True) as tape:
            mean, logvar = model.encode(x)
            z = model.reparameterize(mean, logvar)
            x_logit = model.decode(z,args.apply_sigmoid)
            class_labels=classifier(x_logit)
            loss=compute_creativity_loss(class_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        creativity_loss(loss)
        return loss

    @tf.function
    def distributed_creativity_step(x):
        per_replica_losses = strategy.run(creatvity_step, args= (x,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)
        


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

@tf.function
def distributed_resnet_style_step(x):
    per_replica_losses = strategy.run(resnet_style_step, args=(x,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)


def generate_and_save_images(model, epoch, test_sample,apply_sigmoid):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    print("z shape ",z.shape)
    predictions = model.sample(z,apply_sigmoid)
    print("real prediction shape",predictions.shape)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')

    plt.savefig('{}/{}/image_at_epoch_{:04d}.png'.format(gen_img_dir,args.name,epoch))
    plt.show()

def generate_from_noise(model,epoch,random_vector,apply_sigmoid):
    predictions = model.sample(random_vector,apply_sigmoid)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')

    plt.savefig('{}/{}/gen_image_at_epoch_{:04d}.png'.format(gen_img_dir,args.name,epoch))
    plt.show()

loader=get_loader(args.max_dim,args.genres,args.quantity,root_dict[args.dataset],not args.use_smote)

print("data set length",len([l for l in loader]))

test_dataset = loader.enumerate().filter(lambda x,y: x % args.test_split == 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

train_dataset = loader.enumerate().filter(lambda x,y: x % args.test_split != 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

# Pick a sample of the test set for generating output images
assert args.batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    break

train_dataset=strategy.experimental_distribute_dataset(train_dataset)
test_dataset=strategy.experimental_distribute_dataset(test_dataset)

generate_and_save_images(model, 0, test_sample,False)
generate_from_noise(model,0,random_vector_for_generation,False)

if args.fid:
    fid_score=fid_step(model,random_vector_fid,loader)
    print("FID score {}".format(fid_score))
    with fid_summary_writer.as_default():
        tf.summary.scalar("fid_score",fid_score,step=0)

for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        distributed_train_step(train_x)
        if args.vgg_style:
            distributed_vgg_style_step(train_x)
        if args.resnet:
            distributed_resnet_style_step(train_x)
        if args.gan and args.gan_start<=epoch:
            gen_training = (epoch % args.n_critic== True)
            distributed_adversarial_step(train_x,gen_training)
        if args.creativity and epoch >=args.creativity_start:
            distributed_creativity_step(train_x)
    end_time = time.time()
    with train_summary_writer.as_default():
        tf.summary.scalar('training_loss', train_loss.result(), step=epoch)
    if args.vgg_style:
        with vgg_summary_writer.as_default():
            tf.summary.scalar('vgg_loss',vgg_loss.result(),step=epoch)
    
    if args.resnet:
        with resnet_summary_writer.as_default():
            tf.summary.scalar('resnet_loss',resnet_loss.result(),step=epoch)

    if args.creativity and epoch >=args.creativity_start:
        with creativity_summary_writer.as_default():
            tf.summary.scalar("creativity_loss",creativity_loss.result(),step=epoch)


    if args.diversity:
        for _ in range(args.diversity_batches):
            z=tf.random.normal(shape=[8, args.latent_dim])
            distributed_diversity_step(z)

        with diversity_summary_writer.as_default():
            tf.summary.scalar('diversity_loss',diversity_loss.result(),step=epoch)

    if args.gan:
        with disc_summary_writer.as_default():
            tf.summary.scalar("disc_loss",disc_loss.result(),step=epoch)
        with gen_summary_writer.as_default():
            tf.summary.scalar("gen_loss",gen_loss.result(),step=epoch)
        with gp_summary_writer.as_default():
            tf.summary.scalar("gradient_penalty",gp_loss.result(),step=epoch)

        gp_loss.reset_states()
        disc_loss.reset_states()
        gen_loss.reset_states()

    for test_x in test_dataset:
        distributed_test_step(test_x)
    elbo = -test_loss.result()

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)

    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))
    if epoch %5==0:
        generate_and_save_images(model, epoch, test_sample,args.apply_sigmoid)
        generate_from_noise(model,epoch,random_vector_for_generation,args.apply_sigmoid)

    if epoch %args.fid_interval==0 and args.fid:
        fid_score=fid_step(model,random_vector_fid,loader)
        print("FID score {}".format(fid_score))
        with fid_summary_writer.as_default():
            tf.summary.scalar("fid_score",fid_score,step=epoch)


    train_loss.reset_states()
    test_loss.reset_states()
    diversity_loss.reset_states()
    vgg_loss.reset_states()
    fid_loss.reset_states()
    resnet_loss.reset_states()

if args.fid:
    fid_score=fid_step(model,random_vector_fid,loader)
    print("FID score {}".format(fid_score))
    with fid_summary_writer.as_default():
        tf.summary.scalar("fid_score",fid_score,step=args.epochs + 1)
    fid_loss.reset_states()

if args.gan and args.extra_epochs>0:
    for epoch in range(args.epochs+1,1+args.epochs+args.extra_epochs):
        start_time = time.time()
        for train_x in train_dataset:
            gen_training = (epoch % args.n_critic== True)
            distributed_adversarial_step(train_x,gen_training)
        end_time = time.time()
        print('Epoch: {}, disc loss: {}, time elapse for current epoch: {}'
                .format(epoch, disc_loss.result(), end_time - start_time))
        with disc_summary_writer.as_default():
            tf.summary.scalar("disc_loss",disc_loss.result(),step=epoch)
        with gen_summary_writer.as_default():
            tf.summary.scalar("gen_loss",gen_loss.result(),step=epoch)
        with gp_summary_writer.as_default():
            tf.summary.scalar("gradient_penalty",gp_loss.result(),step=epoch)

        gp_loss.reset_states()
        disc_loss.reset_states()
        gen_loss.reset_states()

        if epoch %5==0:
            generate_and_save_images(model, epoch, test_sample,args.apply_sigmoid)
            generate_from_noise(model,epoch,random_vector_for_generation,args.apply_sigmoid)

        if epoch %args.fid_interval==0 and args.fid:
            fid_score=fid_step(model,random_vector_fid,loader)
            print("FID score {}".format(fid_score))
            with fid_summary_writer.as_default():
                tf.summary.scalar("fid_score",fid_score,step=epoch)
            fid_loss.reset_states()

if args.fid:
    fid_score=fid_step(model,random_vector_fid,loader)
    print("FID score {}".format(fid_score))
    with fid_summary_writer.as_default():
        tf.summary.scalar("fid_score",fid_score,step=epoch)
    fid_loss.reset_states()

checkpoint_path=checkpoint_dir+"/"+args.name
if args.save:
    model.save(checkpoint_path)
    print("saved at ",checkpoint_path)

if args.generate_smote:
    target_root=root_dict[args.dataset]
    target_styles=[]
    for s in os.listdir(target_root):
        path=os.path.join(target_root,s)
        if os.path.isdir(path):
            length=len(get_npz_paths(args.max_dim,[s],target_root,True))
            if length>args.smote_minimum and length < args.smote_maximum:
                target_styles.append(s)
    #target_styles=[s for s in os.listdir(target_root) if s[0]!='.' and len(os.listdir(os.path.join(target_root,s))) >args.smote_minimum and len(os.listdir(os.path.join(target_root,s))) <args.smote_maximum]
    print(target_styles)
    for style in target_styles:
        synthetic=make_smote(model,args.max_dim,style,args.smote_maximum)
        print("style {} with {} SMOTE things".format(s,len(synthetic)))
        for s,img in enumerate(synthetic):
            new_file='{}.{}.{}.npy'.format(smote_sample,s,args.max_dim)
            new_path=os.path.join(target_root,style,new_file)
            np.save(new_path,255*img)
            print("\tsaved at ",new_path)


if args.evaluation_imgs >0:
    import tensorflow_hub as hub
    super_res = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
    eval_dir=args.evaluation_path+args.name
    os.makedirs(eval_dir)
    evaluation_latent_vector = tf.random.normal(
        shape=[args.evaluation_imgs, args.latent_dim])
    predictions = model.sample(evaluation_latent_vector,args.apply_sigmoid)

    predictions=tf.image.resize(predictions, [256,256])

    predictions= super_res(predictions)
    
    plt.figure()

    for i in range(predictions.shape[0]):
        plt.imshow(predictions[i])
        plt.savefig('{}/{}.png'.format(eval_dir,i))
        plt.show()
    

print("all done!")