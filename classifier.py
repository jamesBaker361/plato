import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from data_loader import get_loader_labels
from string_globals import *
import argparse
from vgg import VGGPreProcess, Identity
import random

class RandomAugmentation(tf.keras.layers.Layer):
    def __init__(self,factor,*args,**kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.factor=factor

class RandomHorizontalFlip(RandomAugmentation):
    def call(self,inputs):
        if random.uniform(0,1)<self.factor:
            return tf.image.flip_left_right(inputs)
        return inputs

class RandomAddNoise(RandomAugmentation):
    def call(self,inputs):
        if random.uniform(0,1)<self.factor:
            shape=tf.shape(inputs)
            return inputs + tf.random.normal(shape,0,1.0/255)
        return inputs

class RandomContrast(RandomAugmentation):
    def call(self,inputs):
        if random.uniform(0,1)<self.factor:
            return tf.image.random_contrast(inputs,0.5,1.5)
        return inputs

class RandomBrightness(RandomAugmentation):
    def call(self,inputs):
        if random.uniform(0,1)<self.factor:
            return tf.image.random_brightness(inputs,0.5,1.5)
        return inputs
    

def efficient_cfier(model_level,input_shape,styles,weights="imagenet",activation=True,dropout=False,dense_layers=0,layer_1_n=64,layer_2_n=32,image_rotate=0.0,image_flip=0.0):
    if model_level=="dc":
        return dc_net(input_shape,styles,activation)
    elif model_level=="vgg":
        return vgg_net(input_shape,styles,weights,activation)
    names_to_models={
        "B0":tf.keras.applications.EfficientNetV2B0,
        "B1":tf.keras.applications.EfficientNetV2B1,
        "B2":tf.keras.applications.EfficientNetV2B2,
        "B3":tf.keras.applications.EfficientNetV2B3,
        "L":tf.keras.applications.EfficientNetV2L,
        "M":tf.keras.applications.EfficientNetV2M,
        "S":tf.keras.applications.EfficientNetV2S
    }

    efficient=names_to_models[model_level](include_top=False,weights=weights,input_shape=input_shape,include_preprocessing=False)

    layers=[
        tf.keras.layers.Rescaling(2.0,offset=-1), #scales [0-1] images to [-1,1] images
        efficient,
        tf.keras.layers.Flatten()]

    if dropout:
        layers+=[tf.keras.layers.Dropout(0.2),]

        if dense_layers >0:
            layers+=[tf.keras.layers.Dense(layer_1_n,activation='relu'),
            tf.keras.layers.Dropout(0.2),]

        if dense_layers > 1:
            layers+=[tf.keras.layers.Dense(layer_2_n,activation='relu'),
            tf.keras.layers.Dropout(0.2),]

    layers.append(tf.keras.layers.Dense(len(styles)))

    model= tf.keras.Sequential(layers)

    if activation:
        model.add(tf.keras.layers.Softmax())

    return model

def dc_net(input_shape,styles,activation=True):
    layers=[tf.keras.layers.InputLayer(input_shape=input_shape)]
    width=input_shape[0]
    latent=16
    while width >2:
        layers+=[tf.keras.layers.Conv2D(latent,(3,3),(2,2),padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()]
        width=width/2
        latent=latent*2
    layers+=[tf.keras.layers.Flatten(),tf.keras.layers.Dense(len(styles))]
    if activation:
        layers.append(tf.keras.layers.Softmax())
    return tf.keras.Sequential(layers)

def vgg_net(input_shape,styles,weights="imagenet",activation=True):
    vgg = tf.keras.applications.VGG19(include_top=False, weights=weights,input_shape=input_shape)
    model=tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        VGGPreProcess(),
        vgg,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(styles))
    ])

    if activation:
        model.add(tf.keras.layers.Softmax())

    return model

def get_discriminator(model_level,input_shape,weights,activation=False):
    '''It takes a model level, an input shape, a source of weights (None or imagenet).
    Since we're doing WGAN-GP, the discriminator has a linear activation function
    
    Parameters
    ----------
    model_level
        the level of the model, 
    input_shape
        the shape of the input image
    weights
        where to get weights from
    
    Returns
    -------
        A model
    
    '''
    print(model_level)
    if model_level=="dc":
        return dc_net(input_shape,["_"],activation=activation)
    elif model_level=="vgg":
        return vgg_net(input_shape,["_"],weights,activation=activation)
    return efficient_cfier(model_level,input_shape,["_"],weights,activation=activation)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='get some args')

    #parser.add_argument("--name",type=str,default="classifier",help="name for logs")
    parser.add_argument("--dataset",type=str,default="faces2",help="name of dataset (mnist or art or faces)")
    parser.add_argument("--logdir",type=str,default='logs/default/')
    parser.add_argument("--quantity",type=int,default=250,help="quantity of images to use for training")
    parser.add_argument("--test_split",type=int,default=10,help="1/test_split will be for testing")
    parser.add_argument("--weights",type=str,default=None,help="weights= imagenet")
    parser.add_argument("--level",type=str,default="S",help="level of efficient net to use")
    parser.add_argument("--save",type=bool,default=False, help="whether to save the model or not")
    parser.add_argument("--epochs",type=int,default=10,help="epochs to train for")
    parser.add_argument("--batch_size",type=int,default=16,help="batch size for training/loading data")
    parser.add_argument("--max_dim",type=int,default=64,help="dimensions of input image")
    parser.add_argument("--lr",type=float,default=0.01,help="learning rate for optimizer")
    parser.add_argument("--activation",type=bool,default=False,help="whether to use softmax activation")
    parser.add_argument("--dropout",type=bool,default=False,help="whether to use dropout or not")
    parser.add_argument("--dense_layers",type=int,default=0,help="extra layers for classifier")
    parser.add_argument("--name",type=str,default="")
    parser.add_argument("--layer_1_n",type=int,default=32)
    parser.add_argument("--layer_2_n",type=int,default=16)
    parser.add_argument("--flip_factor",type=float,default=0.0, help="proportion of images to randomly flip")
    parser.add_argument("--noise_factor",type=float,default=0.0,help="proprtion of images to randomly add noise to")
    parser.add_argument("--rotation_factor",type=float,default=0,help="how much to rotate each image")
    parser.add_argument("--brightness_factor",type=float,default=0.0,help="proportion to augment brightness")
    parser.add_argument("--contrast_factor",type=float,default=0.0)
    parser.add_argument("--use_smote",type=bool,default=False,help="whether to use smote (synthetic) data")

    #parser.add_argument("--model_type",type=str,default="efficient",help="")



    args = parser.parse_args()

    def format_classifier_name(dataset=args.dataset,max_dim=args.max_dim,level=args.level,weights=args.weights,lr=args.lr,activation=args.activation):
        if len(args.name)>0:
            return args.name
        return '{}-{}-{}-{}-{}-{}'.format(dataset,max_dim,level,weights,lr,activation)

    styles=dataset_default_all_styles[args.dataset]

    print("# of styles ",len(styles))
    print(styles)

    train_log_dir = args.logdir + format_classifier_name() + '/train'
    test_log_dir = args.logdir + format_classifier_name() + '/test'
    accuracy_log_dir=args.logdir + format_classifier_name() + '/accuracy'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    accuracy_summary_writer=tf.summary.create_file_writer(accuracy_log_dir)

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

    loader=get_loader_labels(args.max_dim,styles,args.quantity,root_dict[args.dataset],not args.use_smote)

    print("data set length",len([l for l in loader]))

    test_dataset = loader.enumerate().filter(lambda x,y: x % args.test_split == 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

    train_dataset = loader.enumerate().filter(lambda x,y: x % args.test_split != 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

    with strategy.scope():
        cfier=efficient_cfier(args.level,(args.max_dim,args.max_dim,3),styles,args.weights,args.activation,args.dropout,args.dense_layers,args.layer_1_n,args.layer_2_n)
        augment_layers=[
            Identity(),
        tf.keras.layers.RandomRotation(args.rotation_factor),
        RandomHorizontalFlip(args.flip_factor),
        RandomAddNoise(args.noise_factor),
        RandomBrightness(args.brightness_factor),
        RandomContrast(args.contrast_factor)]

        augment_model=tf.keras.Sequential(augment_layers)

        optimizer=tf.keras.optimizers.Adam(args.lr)

        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        accuracy=tf.keras.metrics.Mean("accuracy",dtype=tf.float32)

        categorical_cross_entropy=tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        acc_metric=tf.keras.metrics.CategoricalAccuracy()

        def classification_loss(labels,predicted_labels):
            loss=categorical_cross_entropy(labels,predicted_labels)
            return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

    #end strategy.scope

    def train_step(images,labels):
        images=augment_model(images)
        with tf.GradientTape() as tape:
            predicted_labels=cfier(images)
            loss=classification_loss(labels,predicted_labels)
        gradients = tape.gradient(loss, cfier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cfier.trainable_variables))
        train_loss(loss)
        return loss

    def test_step(images,labels):
        predicted_labels=cfier(images)
        loss=classification_loss(labels,predicted_labels)
        acc_metric.update_state(labels,predicted_labels)
        accuracy(acc_metric.result())
        test_loss(loss)
        return loss

    @tf.function
    def distributed_train_step(images,labels):
        per_replica_losses = strategy.run(train_step, args=(images,labels,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                            axis=None)

    @tf.function
    def distributed_test_step(images,labels):
        return strategy.run(test_step, args=(images,labels,))

    train_dataset=strategy.experimental_distribute_dataset(train_dataset)
    test_dataset=strategy.experimental_distribute_dataset(test_dataset)

    for epoch in range(args.epochs):
        for (images,labels) in train_dataset:
            distributed_train_step(images,labels)

        with train_summary_writer.as_default():
            tf.summary.scalar('training_loss', train_loss.result(), step=epoch)

        for (images,labels) in test_dataset:
            distributed_test_step(images,labels)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)

        with accuracy_summary_writer.as_default():
            tf.summary.scalar("accuracy",accuracy.result(),step=epoch)

        print('Epoch: {}, test loss: {}'.format(epoch,test_loss.result()))

        train_loss.reset_states()
        test_loss.reset_states()
        accuracy.reset_states()

    checkpoint_path=checkpoint_dir+"/"+format_classifier_name()
    if args.save:
        cfier.save(checkpoint_path)
        print("saved at ",checkpoint_path)