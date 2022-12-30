import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from cvae import CVAE
from data_loader import *
from fid import calculate_fid
import logging
from smote_data import *
from data_helper import *
from generate_img_helpers import *
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

def get_steps(model,
    strategy,
    args, 
    optimizer, 
    train_loss, 
    compute_loss, 
    test_loss, 
    validate_loss, 
    compute_diversity_loss,
    diversity_loss,
    compute_vgg_style_loss,
    vgg_loss,
    compute_resnet_style_loss,
    resnet_loss,
    disc,
    compute_gan_loss,
    compute_gradient_penalty,
    gen_loss,
    disc_loss,
    gp_loss,
    disc_optimizer,
    classifier,
    compute_creativity_loss,
    creativity_loss
    ):
    def train_step(x):
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

    def validate_step(x):
        loss=compute_loss(x)
        validate_loss(loss)
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

    def fid_step(model,noise,images2):
        loss=0.0
        if args.c3vae:
            for g in range(len(args.genres)):
                indices=[g for _ in range(len(noise))]
                class_label=tf.one_hot(indices,len(args.genres))
                images_real=[]
                print(tf.shape(images2[0]),tf.shape(images2[1]))
                for i in range(len(images2[0])):
                    #print('images2[1][i]',images2[1][i])
                    #print('tf.one_hot(g,len(args.genres)',tf.one_hot(g,len(args.genres)))
                    if tf.math.reduce_all(tf.equal(images2[1][i],tf.one_hot(g,len(args.genres)))).numpy():
                        images_real.append(images2[0][i])
                print('len(images_real)',len(images_real))
                images1=model.sample(noise, False, class_label)
                loss+=calculate_fid(images1, images_real,(args.max_dim,args.max_dim,3))
            loss/=len(args.genres)              
        else:
            images1=model.sample(noise,False)
            loss=calculate_fid(images1,images2,(args.max_dim,args.max_dim,3))
        return loss


    def adversarial_step(x,gen_training=False):
        '''> We sample a batch of random noise, pass it through the generator, and then pass the generated
        images and real images through the discriminator. 
        
        Parameters
        ----------
        x
            The real images input to discriminator
        gen_training, optional
            Whether to train the generator or not.
        
        Returns
        -------
            The generator and discriminator losses are being returned.
        
        '''
        z=tf.random.truncated_normal((args.batch_size,args.latent_dim))
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            fake_x=model.decode(z,args.apply_sigmoid)

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

    def creatvity_step(x):
        with tf.GradientTape(persistent=True) as tape:
            if args.c3vae:
                [x,x_class]=x
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,x_class,args.apply_sigmoid)
            else:
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
    def distributed_validate_step(x):
        return strategy.run(validate_step,args=(x,))

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

    return distributed_adversarial_step, distributed_creativity_step, distributed_diversity_step,distributed_resnet_style_step,distributed_test_step,distributed_train_step,distributed_validate_step,distributed_vgg_style_step,fid_step