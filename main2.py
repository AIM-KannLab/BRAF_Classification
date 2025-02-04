import os
import numpy as np
import pandas as pd
import seaborn as sn
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import argparse
import random
import tensorflow as tf
from data_generator import train_generator
from data_generator import val_generator
from generate_model import generate_model
from transfer_model import transfer_model
from train import train
from test import test
from test_consensus import test_consensus
from opts import parse_opts
from statistics.get_stats_plots import get_stats_plots








def main(opt):

    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    if opt.proj_dir is not None:
        out_dir = opt.proj_dir + '/output'
        model_dir = opt.proj_dir + '/model'
        log_dir = opt.proj_dir + '/log'
        tumor_cls_dir = opt.proj_dir + '/tumor_cls'
        braf_cls_dir = opt.proj_dir + '/braf_cls'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        print('get correct root dir to start.')

    # data generator for train and val data
    """if opt.load_data:
        # data generator
        train_gen = train_generator(
            task=opt.task,
            tumor_cls_dir=tumor_cls_dir,
            braf_cls_dir=braf_cls_dir,
            batch_size=opt.batch_size,
            channel=opt.channel)
        x_val, y_val, val_gen = val_generator(
            task=opt.task,
            tumor_cls_dir=tumor_cls_dir,
            braf_cls_dir=braf_cls_dir,
            batch_size=opt.batch_size,
            channel=opt.channel)
        print('train and val dataloader sucessful!')"""

    # get CNN model
    #cnns = ['ResNet50', 'EfficientNetB4', 'MobileNet', 
            #'DenseNet121', 'IncepttionV3', 'VGG16']
    cnns = ['ResNet50']
    for opt.cnn_model in cnns:
        if opt.transfer_learning:
            my_model = transfer_model(
                cnn_model=opt.cnn_model,
                input_shape=opt.input_shape,
                activation=opt.activation,
                freeze_layer=opt.freeze_layer,
                model_dir=model_dir,
                trained_weights=opt.trained_weights,
                saved_model=opt.saved_model,
                tune_step=opt.tune_step,
                loss_function=opt.loss_function,
                lr=opt.lr)
        else:
            my_model = generate_model(
                cnn_model=opt.cnn_model,
                input_shape=opt.input_shape,
                activation=opt.activation)

        # train model
        if opt.train:
            final_model = train(
                root_dir=opt.proj_dir,
                out_dir=out_dir,
                log_dir=log_dir,
                model_dir=model_dir,
                model=my_model,
                cnn_model=opt.cnn_model,
                train_gen=train_gen,
                val_gen=val_gen,
                x_val=x_val,
                y_val=y_val,
                batch_size=opt.batch_size,
                epoch=opt.epoch,
                loss_function=opt.loss_function,
                lr=opt.lr,
                task=opt.task,
                freeze_layer=opt.freeze_layer,
                trained_weights=opt.trained_weights,
                transfer_learning= opt.transfer_learning)
            print('training complete!')

    # test model
    if opt.test:
        loss, acc = test(
            task=opt.task,
            model=my_model,
            run_type=opt.run_type, 
            channel=opt.channel,
            subtype=opt.subtype,
            model_dir=model_dir, 
            pro_data_dir=opt.pro_data_dir, 
            saved_model=opt.saved_model,
            lr=opt.lr,
            loss_function=opt.loss_function,
            threshold=opt.thr_img, 
            activation=opt.activation,
            _load_model=opt._load_model)
        print('testing complete!')
    
    # get stats and plots
    if opt.stats_plots:
        get_stats_plots(
            task=opt.task,
            channel=opt.channel,
            pro_data_dir=opt.pro_data_dir,
            root_dir=opt.proj_dir,
            run_type=opt.run_type,
            run_model=opt.cnn_model,
            loss=None,
            acc=None,
            saved_model=opt.saved_model,
            epoch=opt.epoch,
            batch_size=opt.batch_size,
            lr=opt.lr,
            thr_img=opt.thr_img,
            thr_prob=opt.thr_prob,
            thr_pos=opt.thr_pos,
            bootstrap=opt.n_bootstrap)
    if opt.test_consensus:
        test_consensus(
            task=opt.task,
            model1=my_model,
            model2=my_model,
            model3=my_model,
            run_type=opt.run_type, 
            channel=opt.channel,
            model_dir=model_dir, 
            pro_data_dir=opt.pro_data_dir, 
            saved_model=opt.saved_model,
            lr=opt.lr,
            loss_function=opt.loss_function,
            threshold=opt.thr_img, 
            activation=opt.activation,
            _load_model=opt._load_model)
        print('testing complete!')
    


if __name__ == '__main__':

    opt = parse_opts()

    main(opt)




