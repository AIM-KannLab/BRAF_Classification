#!/bin/bash

conda activate 2d_approach 

## get the min and max tumor index from segmentation  
python get_min_max.py --image "${PWD}/aidan_segmentation/nnUNet_pLGG/output_preprocess/nnunet/imagesTs/input_0000.nii.gz" --mask "${PWD}/aidan_segmentation/nnUNet_pLGG/output_mask/input_t2w_mri.nii.gz"
## get the sliced data 
python pLGG/get_BRAF_data_v2.py

## Infer the models 
# infer wildtype classifier
python pLGG/main2.py --saved_model tumor__wildtype_radimagenet_fusion_crosstrain_fullimage_internaltestasvalidationResNet50_imagenet_23_0.73.h5 --subtype wildtype
# infer fusion classifier 
python pLGG/main2.py --saved_model tumor_fusion_radimagenet_fullimage_internaltestasvalidationResNet50_imagenet_21_0.75.h5 --subtype fusion
# infer v600e classifier
python pLGG/main2.py --saved_model tumor_v600e_radimagenet_wildtypecrosstrain_filteredv600e_fullimage_internaltestasvalidationResNet50__35_0.73.h5 --subtype v600e
## run the consensus decision block script 
python consensus.py
## output the classification decision 
python pLGG/decision.py
