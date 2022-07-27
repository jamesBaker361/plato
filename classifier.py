import tensorflow as tf
from data_loader import get_loader_labels
from string_globals import *
import argparse

parser = argparse.ArgumentParser(description='get some args')

#parser.add_argument("--name",type=str,default="classifier",help="name for logs")
parser.add_argument("--dataset",type=str,default="mnist",help="name of dataset (mnist or art or faces)")
parser.add_argument("--logdir",type=str,default='logs/gradient_tape/')
parser.add_argument("--quantity",type=int,default=1000,help="quantity of images to use for training")
parser.add_argument("--test_split",type=int,default=10,help="1/test_split will be for testing")
parser.add_argument("--weights",type=str,default=None,help="weights= imagenet")
parser.add_argument("--level",type=str,default="B0",help="level of efficient net to use")
parser.add_argument("--save",type=bool,default=False, help="whether to save the model or not")
parser.add_argument("--epochs",type=int,default=10,help="epochs to train for")
parser.add_argument("--batch_size",type=int,default=16,help="batch size for training/loading data")
parser.add_argument("--max_dim",type=int,default=64,help="dimensions of input image")
#parser.add_argument("--model_type",type=str,default="efficient",help="")



args = parser.parse_args()

def format_classifier_name(dataset=args.dataset,max_dim=args.max_dim,level=args.level,weights=args.weights):
    return '{}-{}-{}-{}'.format(dataset,max_dim,level,weights)

styles=dataset_default_all_styles[args.dataset]



def efficient_cfier(model_level,input_shape,styles,weights="imagenet"):
    names_to_models={
        "B0":tf.keras.applications.EfficientNetB0,
        "B1":tf.keras.applications.EfficientNetB1,
        "B2":tf.keras.applications.EfficientNetB2,
        "B3":tf.keras.applications.EfficientNetB3,
        "B4":tf.keras.applications.EfficientNetB4,
        "B5":tf.keras.applications.EfficientNetB5,
        "B6":tf.keras.applications.EfficientNetB6,
        "B7":tf.keras.applications.EfficientNetB7
    }

    efficient=names_to_models[model_level](include_top=False,weights=weights,input_shape=input_shape)

    return tf.keras.Sequential([
        efficient,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(styles),activation=tf.nn.softmax)
    ])

def discriminator(model_level,input_shape,weights):
    return efficient_cfier(model_level,input_shape,[True],weights)

if __name__=="__main__":

    train_log_dir = args.logdir + format_classifier_name() + '/train'
    test_log_dir = args.logdir + format_classifier_name() + '/test'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

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

    loader=get_loader_labels(args.max_dim,styles,args.quantity,root_dict[args.dataset])

    test_dataset = loader.enumerate().filter(lambda x,y: x % args.test_split == 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

    train_dataset = loader.enumerate().filter(lambda x,y: x % args.test_split != 0).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

    with strategy.scope():
        cfier=efficient_cfier(args.level,(args.max_dim,args.max_dim,3),styles,args.weights)
        optimizer=tf.keras.optimizers.Adam(0.01)

        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

        categorical_cross_entropy=tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

        def classification_loss(labels,predicted_labels):
            loss=categorical_cross_entropy(labels,predicted_labels)
            return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

    #end strategy.scope

    def train_step(images,labels):
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

        print('Epoch: {}, test loss: {}'.format(epoch,test_loss.result()))

        train_loss.reset_states()
        test_loss.reset_states()

    checkpoint_path=checkpoint_dir+"/"+format_classifier_name()
    if args.save:
        cfier.save(checkpoint_path)