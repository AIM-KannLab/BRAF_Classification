import argparse
import os 

def parse_opts():
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument('--proj_dir', default=os.environ.get("PWD")+'/BRAF', type=str, help='Root path')
    
    # data preprocessing
    parser.add_argument('--manual_seed', default=1234, type=int, help='seed')
    parser.add_argument('--channel', default=3, type=int, help='Input channel (3|1)')
    parser.add_argument('--load_data', action='store_true', help='If true, load data is performed.')
    parser.set_defaults(load_data=True)
    parser.add_argument('--save_nii', action='store_true', help='If true, load data is performed.')
    parser.set_defaults(save_nii=False)
    
    # train model
    parser.add_argument('--task', default='tumor', type=str, help='BRAF|tumor')
    parser.add_argument('--run_type', default='test', type=str, help='train|val|test')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epoch', default=50, type=int, help='Epoch')
    parser.add_argument('--activation', default='sigmoid', type=str, help='Activation function on last layer')
    parser.add_argument('--loss_function',  default='binary_crossentropy', type=str, help='loss function')
    parser.add_argument('--cnn_model', default='ResNet50', type=str, help='cnn model')
    parser.add_argument('--input_shape', default=(192, 192, 3), type=int, help='Input shape')
    
    # test model   
    parser.add_argument('--thr_img', default=0.5, type=float, help='threshold to decide class on image level')
    parser.add_argument('--thr_prob', default=0.5, type=float, help='threshold to decide class on patient level')
    parser.add_argument('--thr_pos', default=0.5, type=float, help='threshold to decide class on patient level')
    parser.add_argument('--subtype', default='wildtype', type=str, help='individual subtype classifier')
    parser.add_argument('--n_bootstrap', default=1000, type=int, help='n times of bootstrap to calcualte 95% CI')
    parser.add_argument('--_load_model', default='load_weights', type=str, help='load_model|load_weights')
    parser.add_argument('--saved_model', default='tumor_v600e_radimagenet_wildtypecrosstrain_filteredv600e_fullimage_internaltestasvalidationResNet50__35_0.73.h5', type=str, help='saved model name')
    parser.add_argument('--pro_data_dir', default=os.environ.get("PWD")+'/BRAF/2d_data', type=str, help='saved model name')

    # fine tune model
    parser.add_argument('--trained_weights', default='tumor_v600e_radimagenet_wildtypecrosstrain_filteredv600e_fullimage_internaltestasvalidationResNet50__35_0.73.h5', type=str, help='weights fine tuning')
    parser.add_argument('--freeze_layer', default=12, type=str, help='Freeze layer to train')
    parser.add_argument('--tune_step', default='fine_tune', type=str, help='pre_train|fine_tune')

    # actions
    parser.add_argument('--transfer_learning', action='store_true', help='If true, training is performed.')
    parser.set_defaults(transfer_learning= False)
    parser.add_argument('--train', action='store_true', help='If true, training is performed.')
    parser.set_defaults(train=False)
    parser.add_argument('--test', action='store_true', help='If true, validation is performed.')
    parser.set_defaults(test=True)
    parser.add_argument('--stats_plots', action='store_true', help='If true, plots and statistics is performed.')
    parser.set_defaults(stats_plots=False)
    parser.add_argument('--test_consensus', action='store_true', help='If true, plots and statistics is performed.')
    parser.set_defaults(test_consensus=False)

    args = parser.parse_args()

    return args

    ## python main2.py --saved_model tumor_v600e_radimagenet_wildtypecrosstrain_filteredv600e_fullimage_internaltestasvalidationResNet50__35_0.73.h5 --subtype v600e





