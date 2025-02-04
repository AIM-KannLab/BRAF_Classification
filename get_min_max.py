import os
import numpy as np 
import nibabel as nib 
import pandas as pd 
import argparse

def get_top_bottom_slice_index(mask_image_path):
    mask = nib.load(mask_image_path).get_fdata()
    index = np.where(mask != 0)
    return np.min(index[2]), np.max(index[2])

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Get top and bottom slice indices from a mask file')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file')
    parser.add_argument('--mask', type=str, required=True, help='Path to the mask file')
    
    # Parse arguments
    args = parser.parse_args()
    
    metadata_dict = {
        "bch mrn": [],
        "label": [],
        "top z index": [],
        "bottom z index": [],
        "image path": [],
        "mask path": [],
    }

    bottom_z, top_z = get_top_bottom_slice_index(args.mask)

    metadata_dict["bch mrn"].append(0000000)
    metadata_dict["label"].append(0)
    metadata_dict["top z index"].append(top_z)
    metadata_dict["bottom z index"].append(bottom_z)
    metadata_dict["image path"].append(args.image)
    metadata_dict["mask path"].append(args.mask)

    df = pd.DataFrame(metadata_dict)

    # Save the DataFrame to a CSV file
    csv_file = 'zmin_zmax.csv'
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    main()

