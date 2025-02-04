import pandas as pd 
import os 
import numpy as np 
import csv 


## decision function 

def decision(wildtype_tresh, fusion_tresh, v600e_tresh):

    ## load the three csvs 
    first_stage_wildtype = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/BRAF_mask_external.csv")
    second_stage_fusion = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/FIRST_STAGE_fusion_external.csv")
    third_stage_v600e = pd.read_csv(os.environ.get("PWD")+"/BRAF/2d_data/SECOND_STAGE_v600e_external.csv")

    ## set decision tresholds for each classifier 
    treshold_wildtype = wildtype_tresh
    treshold_fusion = fusion_tresh
    treshold_v600e = v600e_tresh

    ##================================================
    ##          WILD TYPE CHECK 
    ##================================================
    df_mean = first_stage_wildtype.groupby(['ID']).mean().reset_index()
    y_true = df_mean['label'].to_numpy()
    preds = df_mean['y_pred'].to_numpy()
    y_pred = []
    for pred in preds:
        if pred > treshold_wildtype:
            pred = 1
        else:
            pred = 0
        y_pred.append(pred)

    if y_pred[0] == 1:
        return "Wild-type"
    
    ##================================================
    ##          FUSION CHECK 
    ##================================================
    df_mean = second_stage_fusion.groupby(['ID']).mean().reset_index()
    y_true = df_mean['label'].to_numpy()
    preds = df_mean['y_pred'].to_numpy()
    y_pred = []
    for pred in preds:
        if pred > treshold_fusion:
            pred = 1
        else:
            pred = 0
        y_pred.append(pred)

    if y_pred[0] == 1:
        return "Fusion"
    else:
        return "V600E"
    

    """##================================================
    ##          V600E CHECK 
    ##================================================
    df_mean = first_stage_wildtype.groupby(['ID']).mean().reset_index()
    y_true = df_mean['label'].to_numpy()
    preds = df_mean['y_pred'].to_numpy()
    y_pred = []
    for pred in preds:
        if pred > treshold_wildtype:
            pred = 1
        else:
            pred = 0
        y_pred.append(pred)

    if y_pred[0] == 1:
        return "Wild-type"


    


"""
     

if __name__ == "__main__":
    result = decision(wildtype_tresh=0.5, fusion_tresh=0.5, v600e_tresh=0.5)
    print(result)