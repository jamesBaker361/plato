import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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

def oversample(dataset,target_quantity):
    result=[]
    _data=[x for x in dataset]
    while len(result)<target_quantity:
        for x in _data:
            result.append(x)
    return result[:target_quantity]

def get_data_loaders(args,global_batch_size,print_debug,num_examples_to_generate,strategy):

    sq_dict={k:v for k,v in style_quantity_dicts[args.dataset].items() if k in set(args.genres)}

    if args.oversample:
        if args.c3vae:
            _train, _test, _validate=get_loader_oversample_splits(64,sq_dict,args.dataset,labels=True)
        else:
            _train, _test, _validate=get_loader_oversample_splits(64,sq_dict,args.dataset,labels=False)

    else:
        if args.c3vae:
            loader=get_loader_labels(args.max_dim,sq_dict,args.quantity,root_dict[args.dataset],not args.use_smote)
        else:
            loader=get_loader(args.max_dim,args.genres,args.quantity,root_dict[args.dataset],not args.use_smote)

        _test=loader.enumerate().filter(lambda x,y: x % 10== 0).map(lambda x,y: y)

        _train=loader.enumerate().filter(lambda x,y: x % 10 >1).map(lambda x,y: y)

        _validate = loader.enumerate().filter(lambda x,y: x % 10 == 1).map(lambda x,y: y)

    if args.c3vae:
        _train=_train.map(lambda x,y: (tf.cast(x, tf.float32),tf.cast(y, tf.float32)))
        _test=_test.map(lambda x,y: (tf.cast(x, tf.float32),tf.cast(y, tf.float32)))
        _validate=_validate.map(lambda x,y: (tf.cast(x, tf.float32),tf.cast(y, tf.float32)))
    else:
        _train=_train.map(lambda x: tf.cast(x, tf.float32))
        _test=_test.map(lambda x: tf.cast(x,tf.float32))
        _validate=_validate.map(lambda x: tf.cast(x,tf.float32))

    test_dataset = _test.batch(global_batch_size,drop_remainder=True)
    
    validate_dataset = _validate.batch(global_batch_size,drop_remainder=True)

    train_dataset = _train.batch(global_batch_size,drop_remainder=True)

    print_debug("test cardinality ",len([_ for _ in test_dataset]))
    
    print_debug("train cardinality ",len([_ for _ in train_dataset]))

    print_debug("dataset element_spec", train_dataset.element_spec)

    for i in _test.shuffle(100000,seed=123).batch(args.fid_sample_size):
        fid_test_sample=i
        break

    for i in _validate.shuffle(100000, seed=123).batch(args.fid_sample_size):
        fid_validate_sample = i
        break

    # Pick a sample of the test set for generating output images
    assert args.batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
        print_debug('num_examples to generate ', num_examples_to_generate)
        if args.c3vae:
            test_sample = [test_batch[0][0:num_examples_to_generate, :, :, :],test_batch[1][0:num_examples_to_generate]]
        else:
            test_sample = test_batch[0:num_examples_to_generate, :, :, :]
        break

    for validate_batch in validate_dataset.take(1):
        if args.c3vae:
            validate_sample=[validate_batch[0][0:num_examples_to_generate, :, :, :],validate_batch[1][0:num_examples_to_generate]]
        else:
            validate_sample=validate_batch[0:num_examples_to_generate, :, :, :]
        break
    train_dataset=strategy.experimental_distribute_dataset(train_dataset)
    test_dataset=strategy.experimental_distribute_dataset(test_dataset)
    validate_dataset=strategy.experimental_distribute_dataset(validate_dataset)

    return train_dataset,test_dataset,validate_dataset,test_sample,validate_sample,fid_test_sample, fid_validate_sample