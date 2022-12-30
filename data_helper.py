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

def get_data_loaders(args,global_batch_size,print_debug,num_examples_to_generate,strategy):

    sq_dict={k:v for k,v in style_quantity_dicts[args.dataset].items() if k in set(args.genres)}

    if args.oversample and args.c3vae:
        loader=get_loader_oversample_labels(args.max_dim,sq_dict,root_dict[args.dataset])
    elif args.oversample:
        loader=get_loader_oversample(args.max_dim,sq_dict,root_dict[args.dataset])
    elif args.c3vae:
        loader=get_loader_labels(args.max_dim,sq_dict,args.quantity,root_dict[args.dataset],not args.use_smote)
    else:
        loader=get_loader(args.max_dim,args.genres,args.quantity,root_dict[args.dataset],not args.use_smote)

    _test=loader.enumerate().filter(lambda x,y: x % 10== 0).map(lambda x,y: y)

    test_dataset = _test.shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

    _validate = loader.enumerate().filter(lambda x,y: x % 10 == 1).map(lambda x,y: y)

    validate_dataset = _validate.shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

    print_debug("test cardinality ",len([_ for _ in test_dataset]))

    train_dataset = loader.enumerate().filter(lambda x,y: x % 10 >1).map(lambda x,y: y).shuffle(10,reshuffle_each_iteration=False).batch(global_batch_size,drop_remainder=True)

    print_debug("train cardinality ",len([_ for _ in train_dataset]))

    print_debug("dataset element_spec", train_dataset.element_spec)

    for i in _test.shuffle(10000,seed=123).batch(args.fid_sample_size):
        fid_test_sample=i
        break

    for i in _validate.shuffle(10000, seed=123).batch(args.fid_sample_size):
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