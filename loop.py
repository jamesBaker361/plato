import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import optuna
import atexit
import json
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
from stylevae import *
from fid import calculate_fid
from vgg import *
from optimizer_config import get_optimizer
import logging
from smote_data import *
from data_helper import *
from generate_img_helpers import *
from  step_helpers import *
logger = logging.getLogger()
old_level = logger.level
from keras.utils.layer_utils import count_params
logger.setLevel(100)

print(datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

import argparse

parser = argparse.ArgumentParser(description='get some args')

parser.add_argument("--name",type=str,default="")
parser.add_argument("--save",type=str,default=False,help="whether to save this model")
parser.add_argument("--load",default=False,type=bool,help="if there already exists a saved model with this name, load it")
parser.add_argument("--loadpath",type=str,default="",help="path to load saved autoencoder from")
parser.add_argument("--dataset",type=str,default="faces3",help="name of dataset (mnist or art or faces or faces2 or faces3 or deep_weeds)")
parser.add_argument("--use_smote",type=bool,default=False,help="whether to use the normal dataset but with added synthetic samples")
parser.add_argument("--oversample",type=bool,default=False,help="whether to use normal images but oversampling the smaller classes")
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

parser.add_argument("--c3vae",type=bool,default=False,help="whether to use c3vae")
parser.add_argument("--class_latent_dim",type=int,default=32)
parser.add_argument("--c3_class_noise_activation",type =str, default="sigmoid")
parser.add_argument("--c3_img_activation",type =str, default="relu")
parser.add_argument("--c3_output_activation",type =str, default="sigmoid")
#class_noise_activation, img_activation, output_activation

parser.add_argument("--attvae", type=bool, default=False, help='whether to use attentional VAE or not')
parser.add_argument("--patch_size", type=int, default=16,help="patch size for vision transformer")
parser.add_argument("--num_layers",type=int,default=3, help="num of attention layers in attention vae")
parser.add_argument("--d_encoder",type=int,default=64)
parser.add_argument("--d_decoder",type=int,default=64)

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
parser.add_argument("--fid_sample_size",type=int,default=750,help="how many images to do each FID scoring")

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
parser.add_argument("--classifier_path",type=str,default="../../../../../scratch/jlb638/plato/checkpoints/cfier_B3",help="path to load classifier from")
parser.add_argument("--creativity_start",type=int,default=0,help="epoch when to start applying creativity loss")

parser.add_argument("--evaluation_imgs",type=int,default=0,help="how many images to generate at the end for evaluation")
parser.add_argument("--evaluation_path",type=str,default="./evaluation/",help="where to save evaluation images")

parser.add_argument("--begin_epoch",type=int,default=0,help="in case we want to start at a later epoch")
parser.add_argument("--debug",type=bool, default=False, help="whether to print out more debug statemtns")
parser.add_argument("--validate",type=bool, default=False, help="whether to use validation set for evaluation")
parser.add_argument("--test", type=bool,default=False,help="whether to use test set for evaluation")

parser.add_argument("--optuna", type=bool, default=False, help='whether to use optuna for hyperparameter search')
parser.add_argument("--optuna_metric", type=str, default="mse",help="metric to optimize optuna for (mse, fid)")
parser.add_argument("--n_trials",type=int,default=100,help="how many trials to run for optuna")

parser.add_argument("--stylevae",default=False, type=bool, help="whether to use stylevae")
parser.add_argument("--sv_fc_layers",default=4,type=int,help='style vae fully connected layers')
parser.add_argument("--sv_fc_dims",default=16, type=int, help="style vae fully connnected layers dimensionality (#nodes)")
parser.add_argument("--sv_fc_activation",default="sigmoid",type=str,help="stylevae fully connected layer activation function")
parser.add_argument("--sv_fc_dropout_rate",type=float,default=0.1,help='stylevase fully connected dropout rate')

for names,default in zip(["batch_size","max_dim","epochs","latent_dim","quantity","diversity_batches","test_split"],[16,64,10,2,250,4,10]):
    parser.add_argument("--{}".format(names),type=int,default=default)

args = parser.parse_args()

def objective(trial):
    if args.optuna:
        activations=['relu', 'linear', 'selu', 'sigmoid', 'swish', 'tanh']
        args.test=True
        if args.attvae:
            args.patch_size=trial.suggest_categorical('patch_size', [4,8,16,32])
            args.num_layers=trial.suggest_int("num_layers",3,6)
            args.d_encoder=trial.suggest_categorical("d_encoder",[16,32,64,128])
            args.d_decoder=trial.suggest_categorical("d_decoder",[16,32,64,128])

        if args.gan:
            args.n_critic=trial.suggest_int("n_critic",1,8)
            args.level=trial.suggest_categorical("discriminator_level",["dc","vgg" ,"efficient"])

        args.init_lr=trial.suggest_categorical("init_lr",[0.00001,0.00005,0.0001])
        args.batch_size=trial.suggest_categorical("batch_size",[8,16,32,64])
        args.latent_dim=trial.suggest_categorical("latent_dim",[8,16,32,64])

        if args.c3vae:
            args.class_latent_dim=trial.suggest_categorical('class_latent_dim',[16,32,64])
            args.c3_class_noise_activation = 'sigmoid' #trial.suggest_categorical('c3_class_noise_activation', activations)
            args.c3_img_activation = 'relu' #trial.suggest_categorical('c3_img_activation', activations)
            args.c3_output_activation = 'sigmoid' #trial.suggest_categorical('c3_output_activation', activations)
            #class_noise_activation, img_activation, output_activation

        if args.stylevae:
            args.sv_fc_layers=trial.suggest_int('sv_fc_layers',4,8)
            args.sv_fc_dims=trial.suggest_categorical('sv_fc_dims',[16,32,64,128])
            args.sv_fc_activation=trial.suggest_categorical('sv_fc_activation',activations)
            args.sv_fc_dropout_rate=trial.suggest_categorical('sv_fc_dropout_rate',[0.0,0.1,0.25,0.5])

    if len(args.genres)==0:
        args.genres=dataset_default_all_styles[args.dataset]

    num_examples_to_generate = 8

    image_size=(args.max_dim,args.max_dim,3)

    current_time = datetime.datetime.now().strftime("%H%M%S")
    if len(args.name)==0:
        args.name=current_time+"_"+str(randrange(0,20))

    os.makedirs(gen_img_dir+"/"+args.name,exist_ok=True)

    physical_devices=tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device,True)
        except  RuntimeError as e:
            print(e)

    logical_gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy()

    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore

    if len(logical_gpus)>0:
        global_batch_size=len(logical_gpus)*args.batch_size
    else:
        global_batch_size=args.batch_size

    def print_debug(msg,*anon_args,**kwargs):
        if args.debug:
            print(msg,*anon_args,**kwargs)

    def print_shape(msg, tensor):
        if args.c3vae:
            print_debug(msg, tf.shape(tensor[0], tf.shape(tensor[1])))
        else:
            print_debug(msg,tf.shape(tensor))

    train_log_dir = args.logdir + args.name + '/train'
    test_log_dir = args.logdir + args.name + '/test'
    validate_log_dir=args.logdir+args.name+'/validate'
    diversity_log_dir=args.logdir + args.name + '/diversity'
    vgg_log_dir=args.logdir+args.name+"/vgg"
    resnet_log_dir=args.logdir+args.name+"/resnet"
    creativity_log_dir=args.logdir+args.name+"/creativity"



    logpx_z_log_dir=args.logdir + args.name + '/logpx_z'
    logpz_log_dir=args.logdir + args.name + '/logpz'
    logqz_x_log_dir=args.logdir + args.name + '/logqz_x'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    validate_summary_writer=tf.summary.create_file_writer(validate_log_dir)
    diversity_summary_writer=tf.summary.create_file_writer(diversity_log_dir)
    vgg_summary_writer=tf.summary.create_file_writer(vgg_log_dir)
    resnet_summary_writer=tf.summary.create_file_writer(resnet_log_dir)
    creativity_summary_writer=tf.summary.create_file_writer(creativity_log_dir)

    if args.fid:
        random_vector_fid=tf.random.normal(shape=[args.fid_sample_size, args.latent_dim])
        if args.test:
            fid_test_log_dir=args.logdir+args.name+"/fid_test"
            fid_test_summary_writer=tf.summary.create_file_writer(fid_test_log_dir)
        if args.validate:
            fid_validate_log_dir=args.logdir+args.name+"/fid_validate"
            fid_validate_summary_writer=tf.summary.create_file_writer(fid_validate_log_dir)

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
        disc=None
        classifier = None
        disc_optimizer = None
        vgg_style_extractor= None
        resnet_style_extractor=None
        model = CVAE(args.latent_dim,args.max_dim)
        if args.c3vae:
            model=C3VAE(len(args.genres),args.class_latent_dim,args.latent_dim,args.max_dim,args.c3_class_noise_activation, args.c3_img_activation, args.c3_output_activation)
        if args.attvae:
            model=AttentionVAE(args.patch_size,args.num_layers,args.d_encoder,args.d_decoder,args.latent_dim,args.max_dim,strategy)
            model(tf.random.uniform((1,args.max_dim,args.max_dim,3)))
        if args.stylevae:
            model=StyleVAE(args.sv_fc_layers,args.sv_fc_dims,args.sv_fc_dropout_rate,args.sv_fc_activation, args.latent_dim,args.max_dim)
            #model(tf.random.uniform((1,args.max_dim,args.max_dim,3)))
        n_params=count_params(model.trainable_weights)
        print("n_params = ", n_params)
        if args.load or len(args.loadpath) != 0:
            model.encoder=tf.keras.models.load_model(checkpoint_dir+"/"+args.loadpath+"/encoder")
            model.decoder=tf.keras.models.load_model(checkpoint_dir+"/"+args.loadpath+"/decoder")
            print("successfully loaded from {}".format(args.loadpath))
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
        validate_loss=tf.keras.metrics.Mean('validate_loss',dtype=tf.float32)
        diversity_loss=tf.keras.metrics.Mean("diversity_loss",dtype=tf.float32)
        vgg_loss=tf.keras.metrics.Mean("vgg_loss",dtype=tf.float32)
        resnet_loss=tf.keras.metrics.Mean("resnet_loss",dtype=tf.float32)
        fid_test_loss=tf.keras.metrics.Mean("fid_test_loss",dtype=tf.float32)
        fid_validate_loss=tf.keras.metrics.Mean("fid_validate_loss",dtype=tf.float32)
        gp_loss=tf.keras.metrics.Mean("gp_loss",dtype=tf.float32)
        disc_loss=tf.keras.metrics.Mean("disc_loss",dtype=tf.float32)
        gen_loss=tf.keras.metrics.Mean("gen_loss",dtype=tf.float32)
        creativity_loss=tf.keras.metrics.Mean("creativity_loss",dtype=tf.float32)

        def compute_loss(x):
            '''> The function takes in a batch of images, encodes them into a latent space, decodes them back into
            the image space, and then computes the loss- this is standard VAE loss
            
            Parameters
            ----------
            x
                the input image
            
            Returns
            -------
                The loss function is being returned.
            
            '''
            if args.c3vae:
                [x,x_class]=x
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,x_class,args.apply_sigmoid)
            else:
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,args.apply_sigmoid)
            cross_ent=(x-x_logit)**2
            reconst = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
            kl= -0.5* args.kl_weight * tf.reduce_sum(1+logvar - mean**2 - tf.exp(logvar),axis=1)

            per_example_loss= reconst+kl

            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        def compute_diversity_loss(samples,z):
            '''It computes the average of the ratio of the L2 norm of the difference between the samples and the L2
            norm of the difference between the latent vectors based off of https://arxiv.org/abs/1901.09024
            
            Parameters
            ----------
            samples
                the generated samples
            z
                the latent space vector
            
            Returns
            -------
                The loss is being returned.
            
            '''
            try:
                batch_size=len(samples)
            except TypeError:
                try:
                    batch_size=samples.shape[0]
                except:
                    batch_size=tf.shape(samples)[0]
            _loss=tf.constant(0.0,dtype=tf.float32) #[-1.0* tf.reduce_mean(tf.square(tf.subtract(samples, samples)))]
            for i in range(batch_size):
                for j in range(i+1,batch_size):
                    _loss+=tf.norm(samples[i]-samples[j])/tf.norm(z[i]-z[j])
            loss=[-_loss*args.lambd]
            return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

        def compute_style_loss(x,extractor,extractor_lambda):
            '''It takes an image, encodes it, decodes it, and then computes the style loss between the original
            image and the decoded image
            
            Parameters
            ----------
            x
                the input image
            extractor
                the style extractor network (vgg, resnet)
            extractor_lambda
                a list of weights for each layer of the style extractor.
            
            Returns
            -------
                The loss function for the style loss.
            
            '''
            if args.c3vae:
                [x,x_class]=x
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,x_class,args.apply_sigmoid)
            else:
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,args.apply_sigmoid)
            x_style=extractor(x)
            x_logit_style=extractor(x_logit)
            cross_ent=(x_style-x_logit_style)**2
            reconst=tf.reduce_sum(extractor_lambda * cross_ent,axis=[1,2,3])
            return tf.nn.compute_average_loss(reconst,global_batch_size=global_batch_size)

        def compute_vgg_style_loss(x):
            '''It takes a tensor and a style extractor, and returns a function that computes the vgg style loss for
            that tensor
            
            Parameters
            ----------
            x
                the input image
            
            Returns
            -------
                The loss function for the style loss.
            
            '''
            return compute_style_loss(x,vgg_style_extractor,args.vgg_lambda)

        def compute_resnet_style_loss(x):
            '''It takes a tensor and returns a function that computes the resnet style loss of that tensor
            
            Parameters
            ----------
            x
                the input image
            
            Returns
            -------
                The function compute_resnet_style_loss is being returned.
            
            '''
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
            '''It takes a batch of real images and a batch of fake images, and then it computes the gradient of the
            discriminator's output with respect to the input, and then it penalizes the discriminator if the
            gradient is too large, this is for WGAN-GP
            
            Parameters
            ----------
            real_x
                The real images
            fake_x
                The output of the generator.
            
            Returns
            -------
                The gradient penalty is being returned.
            
            '''
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
            print('loaded classifier from ',args.classifier_path)

        def compute_creativity_loss(class_labels):
            '''It computes the cross-entropy loss between the predicted class labels and a uniform distribution
            
            Parameters
            ----------
            class_labels
                the output of the classifier, which is a tensor of shape [batch_size, num_classes]
            
            Returns
            -------
                The loss is being returned.
            
            '''
            uniform=tf.fill([args.batch_size,len(args.genres)],1.0/len(args.genres))
            cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            loss=cce(class_labels,uniform)
            return tf.nn.compute_average_loss(args.creativity_lambda * loss, global_batch_size=global_batch_size)

        train_dataset,test_dataset,validate_dataset,test_sample,validate_sample,fid_test_sample, fid_validate_sample=get_data_loaders(args,global_batch_size,print_debug,num_examples_to_generate,strategy)
    #end strategy scope

    distributed_adversarial_step, distributed_creativity_step, distributed_diversity_step,distributed_resnet_style_step,distributed_test_step,distributed_train_step,distributed_validate_step,distributed_vgg_style_step,fid_step = get_steps(
        model, strategy, args, optimizer, train_loss, compute_loss, test_loss, validate_loss, compute_diversity_loss,
        diversity_loss,compute_vgg_style_loss,vgg_loss,compute_resnet_style_loss,resnet_loss,disc,compute_gan_loss,
        compute_gradient_penalty,gen_loss,disc_loss,gp_loss,disc_optimizer,classifier,compute_creativity_loss,creativity_loss)


    generate_and_save_images=get_generate_and_save_images(print_debug,args)
    generate_from_noise=get_generate_from_noise(args,num_examples_to_generate)

    generate_and_save_images(model, args.begin_epoch, test_sample,False)
    generate_from_noise(model,args.begin_epoch,random_vector_for_generation,False)

    mse=0
    test_fid_score=0
    validate_fid_score=0

    def fid_eval_and_store(step,model):
        test_fid_score=0
        validate_fid_score=0
        if args.test:
            test_fid_score=fid_step(model,random_vector_fid,fid_test_sample)
            print("FID score test {}".format(test_fid_score))
            with fid_test_summary_writer.as_default():
                tf.summary.scalar("fid_test_score",test_fid_score,step=step)
        if args.validate:
            validate_fid_score=fid_step(model,random_vector_fid,fid_validate_sample)
            print("FID score validate {}".format(validate_fid_score))
            with fid_validate_summary_writer.as_default():
                tf.summary.scalar("fid_validate_score",validate_fid_score,step=step)
        return test_fid_score,validate_fid_score

    for epoch in range(args.begin_epoch+1, args.epochs + 1):
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

        if args.validate:
            for validate_x in validate_dataset:
                distributed_validate_step(validate_x)
            elbo = -validate_loss.result()
            with validate_summary_writer.as_default():
                tf.summary.scalar('loss', elbo,step=epoch)
        if args.test:
            for test_x in test_dataset:
                distributed_test_step(test_x)
            elbo = -test_loss.result()
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', elbo, step=epoch)

        display.clear_output(wait=False)
        print('Epoch: {}, time elapse for current epoch: {}'
                    .format(epoch,  end_time - start_time))
        if epoch %5==0:
            generate_and_save_images(model, epoch, test_sample,args.apply_sigmoid)
            generate_from_noise(model,epoch,random_vector_for_generation,args.apply_sigmoid)

        if epoch %args.fid_interval==0 and args.fid:
            test_fid_score,validate_fid_score=fid_eval_and_store(epoch,model)


        mse=test_loss.result()

        if args.optuna:
            if args.optuna_metric=='mse':
                intermediate_value=mse
            else:
                intermediate_value=test_fid_score
            trial.report(intermediate_value, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        train_loss.reset_states()
        test_loss.reset_states()
        validate_loss.reset_states()
        diversity_loss.reset_states()
        vgg_loss.reset_states()
        fid_test_loss.reset_states()
        fid_validate_loss.reset_states()
        resnet_loss.reset_states()
    #end epoch

    if args.fid:
        test_fid_score,validate_fid_score=fid_eval_and_store(args.epochs+1,model)

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
                test_fid_score,validate_fid_score=fid_eval_and_store(epoch,model)

    if args.fid:
        test_fid_score,validate_fid_score=fid_eval_and_store(epoch+1,model)

    if args.evaluation_imgs >0:
        eval_dir=args.evaluation_path+args.name
        os.makedirs(eval_dir,exist_ok=True)
        evaluation_latent_vector = tf.random.normal(
            shape=[args.evaluation_imgs, args.latent_dim])
        predictions = model.sample(evaluation_latent_vector,args.apply_sigmoid)

        predictions=tf.image.resize(predictions, [256,256])
        
        plt.figure()

        for i in range(predictions.shape[0]):
            plt.imshow(predictions[i])
            plt.savefig('{}/{}.png'.format(eval_dir,i))
            plt.show()
        

    if args.save:
        save_path=checkpoint_dir+"/"+args.name
        os.makedirs(save_path,exist_ok=True)
        model.encoder.save(save_path+"/encoder")
        print("saved at ",save_path+"/encoder")
        model.decoder.save(save_path+"/decoder")
        print("saved at ",save_path+"/decoder")

    print("all done!")

    if args.optuna_metric=="mse":
        return mse
    else:
        return test_fid_score

    

if args.optuna:
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    try:
        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    except tf.errors.ResourceExhaustedError:
        print("resources exhausted :(((")
    finally:
        best_params = study.best_params
        print(best_params)
        os.makedirs('./studies/',exist_ok=True)
        with open("./studies/best_{}.json".format(args.name),"w+") as file:
            file.write(json.dumps(best_params))
else:
    objective(None)
