import os
import timeit
import numpy as np
import pandas as pd
import glob
#import skimage.transform as st
from datetime import datetime
from time import gmtime, strftime
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score




def test(task, model, run_type, channel, subtype, model_dir, pro_data_dir, saved_model, lr, loss_function,
         threshold=0.5, activation='sigmoid', _load_model='load_weights'):    
    
    """
    Evaluate model for validation/test/external validation data;

    Args:
        out_dir {path} -- path to main output folder;
        proj_dir {path} -- path to main project folder;
        saved_model {str} -- saved model name;
        tuned_model {Keras model} -- finetuned model for chest CT;
    
    Keyword args:
        threshold {float} -- threshold to determine postive class;
        activation {str or function} -- activation function, default: 'sigmoid';
    
    Returns:
        training accuracy, loss, model
    
    """


    # load data and label based on run type
    if task == 'BRAF_status':
        if run_type == 'val':
            if channel == 1:
                fn_data = 'val_arr_1ch.npy'
            elif channel == 3:
                fn_data = 'val_arr_3ch.npy'
            fn_label = 'val_img_df.csv'
            fn_pred = 'val_img_pred.csv'
        elif run_type == 'test':
            if channel == 1:
                fn_data = 'test_arr_3ch.npy'
            elif channel == 3:
                fn_data = 'test_arr_3ch.npy'
            fn_label = 'test_BRAF_df.csv'
            fn_pred = 'test_BRAF_pred.csv' 
    if task == 'BRAF_fusion':
        if run_type == 'val':
            if channel == 1:
                fn_data = 'val_arr_1ch_.npy'
            elif channel == 3:
                fn_data = 'val_arr_3ch_.npy'
            fn_label = 'val_img_df_.csv'
            fn_pred = 'val_img_pred_.csv'
        elif run_type == 'test':
            if channel == 1:
                fn_data = 'test_arr_3ch_.npy'
            elif channel == 3:
                fn_data = 'test_arr_3ch_.npy'
            fn_label = 'test_fusion_df.csv'
            fn_pred = 'test_fusion_pred.csv'
    if task == 'tumor':
        if run_type == 'val':
            if channel == 1:
                fn_data = '_val_arr_1ch.npy'
            elif channel == 3:
                fn_data = '_val_arr_3ch.npy'
            fn_label = '_val_img_df.csv'
            fn_pred = '_val_img_pred.csv'
        elif run_type == 'test':
            if channel == 1:
                fn_data = 'tx_arr_1ch.npy'
            elif channel == 3:
                fn_data = 'img_tr_arr_3ch.npy'
            fn_label = 'tr_img_df.csv'   
            fn_pred = str(subtype) + '_classifier.csv'
    x_data = np.load(os.path.join(pro_data_dir, fn_data))
    df = pd.read_csv(os.path.join(pro_data_dir, fn_label))
    y_label = np.asarray(df['label']).astype('int').reshape((-1, 1))

    # load saved model and evaluate
    if _load_model == 'load_model':
        model = load_model(os.path.join(model_dir, saved_model))
    elif _load_model == 'load_weights':    # model compile
        auc = tf.keras.metrics.AUC()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=loss_function,
            metrics=["accuracy"])
        model.load_weights(os.path.join(model_dir, task + "/" + saved_model))
    model.load_weights(os.path.join(model_dir, task + "/" + saved_model))
    model.trainable = False
    
    y_pred = model.predict(x_data)
    score = model.evaluate(x_data, y_label)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('loss:', loss)
    print('acc:', acc)
    #auc = roc_auc_score(y_label, y_pred)
    #auc = np.around(auc, 3)
    #print('auc:', auc)
    
    if activation == 'sigmoid':
        y_pred = model.predict(x_data)
        y_pred_class = [1 * (x[0] >= threshold) for x in y_pred]
    elif activation == 'softmax':
        y_pred_prob = model.predict(x_data)
        y_pred = y_pred_prob[:, 1]
        y_pred_class = np.argmax(y_pred_prob, axis=1)

    # save a dataframe
    ID = []
    for file in df['fn']:
        if run_type in ['val', 'test', 'tune']:
            id = file.split('\\')[-1].split('_')[0].strip()
        elif run_type == 'tune2':
            id = file.split('\\')[-1].split('_s')[0].strip()
        ID.append(id)
    df['ID'] = ID
    df['y_pred'] = y_pred
    df['y_pred_class'] = y_pred_class
    df_test_pred = df[['ID', 'label', 'y_pred', 'y_pred_class']]
    df_test_pred.to_csv(os.path.join(pro_data_dir, fn_pred)) 
    
    return loss, acc

        



    

    
