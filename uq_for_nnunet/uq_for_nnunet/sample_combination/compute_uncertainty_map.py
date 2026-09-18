#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Description:    Compute uncertainty maps (e.g. entropy, mutual information, variance)
                from multiple segmentation predictions (e.g., MC Dropout).

Usage:
    python compute_uncertainty_map.py --folder /path/to/npz_files \
                                  --patients 001 002 003 \
                                  --output_dir /path/to/save/uncertainty_maps \
                                  --classes 20

Arguments:
    --folder       Folder containing .npz prediction files
    --patients     List of patient IDs to process (default: all in folder)
    --output_dir   Folder to save the uncertainty maps
    --classes      Number of predicted classes for entropy computations

===============================================================================
"""

import os
import re
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
from uq_for_nnunet.utils.uncertainty_metrics.metrics import entropy_prob, mutual_information_prob, entropy_classwise_prob, mutual_information_classwise_prob, variance_prob, variance_classwise_prob
import nibabel as nib

def setup_logging(output_dir: str) -> str:
    """
    Set up logging to print to console and save to a timestamped text file.

    Args:
        output_dir (str): Directory where the log file will be saved.

    Returns:
        str: Path to the log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"log_compute_uncertainty_map_{timestamp}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging started. Log file: {log_file}")
    return log_file


def save_uncertainty_map_as_nifti(uncertainty_map: np.ndarray, output_file: str, affine: np.ndarray = np.eye(4), roi_dict: dict = None):
    print(f"Shape of the uncertainty map is {np.shape(uncertainty_map)}")
    if len(np.shape(uncertainty_map)) > 3: # save each ROI independently
        print(f'Making uncertainty map for {uncertainty_map.shape[0]} classes')
        for c in range(uncertainty_map.shape[0]):
            print(f'Shape of data is {np.shape(uncertainty_map)}')
            class_map = uncertainty_map[c,:,:,:] #shape: [x,y,z]
            if roi_dict:
                roi_name = roi_dict[c]
                out_file = output_file.replace(".nii.gz", f"_{roi_name}.nii.gz")
            
            class_map = np.transpose(class_map, (1, 2, 0)) #transform
            class_map = np.rot90(class_map, k=1) #transform
            class_map = np.flipud(class_map) #transform
            nifti_img = nib.Nifti1Image(class_map, affine)
            nib.save(nifti_img, out_file)
                
    else:
        data = np.transpose(uncertainty_map, (1, 2, 0)) #transform
        data = np.rot90(uncertainty_map, k=1) #transform
        data = np.flipud(uncertainty_map) #transform
        nifti_img = nib.Nifti1Image(uncertainty_map, affine)
        nib.save(nifti_img, output_file)
    return

def create_uncertainty_map_from_samples(folder: str, patients: list = None, output_dir: str = None, classes: int = 0, keep_classes: list = None, metrics: list = ['entropy'], methods: list = [], roi_dict: str = None):

    setup_logging(output_dir)
    
    if roi_dict:
        import runpy
        # assume roi_dict is a path to a .py file that defines ROI_DICT
        module_vars = runpy.run_path(roi_dict)
        if 'ROI_DICT' not in module_vars:
            raise ImportError(f"Provided roi_dict file {roi_dict} does not define ROI_DICT")
        roi_dict_data = module_vars['ROI_DICT']
        print(roi_dict_data)
        
    for patient in tqdm(patients, desc="Patients"):
        print(f'evaluating patient {patient}')
        for method in methods:
            if method=="mc_dropout":
                pattern = re.compile(f'.*_{patient}_(0?[1-9]|1[0-9]|20)_.npz')
                all_files = [f for f in os.listdir(folder) if pattern.match(f)]
            elif method=="deep_ensemble":
                all_files = []
                for subfolder in os.listdir(folder):
                    if 'fold' in subfolder:
                        print(subfolder)
                        pattern = re.compile(f'.*_{patient}.npz')
                        subfolder_path = os.path.join(folder, subfolder)
                        patient_file = [f for f in os.listdir(subfolder_path) if pattern.match(f)]
                        print(f'Added {os.path.join(subfolder_path, patient_file[0])} \n')
                        all_files.append(os.path.join(subfolder_path, patient_file[0]))
            elif method=="tta":
                all_files = []
                for subfolder in os.listdir(folder):
                    if 'fold' in subfolder:
                        print(subfolder)
                        pattern = re.compile(f'.*_{patient}.npz')
                        subfolder_path = os.path.join(folder, subfolder)
                        patient_file = [f for f in os.listdir(subfolder_path) if pattern.match(f)]
                        print(f'Added {os.path.join(subfolder_path, patient_file[0])} \n')
                        all_files.append(os.path.join(subfolder_path, patient_file[0]))
            else:
                pattern = re.compile(f'.*_{patient}_.npz')
                all_files = [f for f in os.listdir(folder) if pattern.match(f)]
 
            print(all_files)
            

            combined_data = {}
            for file in tqdm(all_files, desc="Samples"):
                data = np.load(os.path.join(folder,file))
                available_classes = np.shape(data['probabilities'])[0]
                print(f"Number of classes available in uncertainty maps is {available_classes} and number of classes specified is {classes}")
                
                if keep_classes is not None:
                    keep_classes = list(map(int, keep_classes)) #convert string elements in list to integers
                   
                    if len(keep_classes) <= available_classes:                      
                        print(f"Shape of data before: {np.shape(data['probabilities'])}")
                        if not 0 in keep_classes: #if background is not mentioned as label to keep add it (as it alwasys SHOULD be used to compute uncertainty maps)
                            keep_classes = [0] + keep_classes
                    if len(keep_classes) != classes:
                        print(f'Number of classes mentioned with arg.classes {classes} not equal to number of classes in keep_classes {len(keep_classes)}')
                        return 
                else:
                    keep_classes = list(range(0, available_classes))                   
                              
                # Apply class filtering
                if keep_classes is not None:
                    print(f' Keeping classes {keep_classes}')
                    data_probabilities = data['probabilities'][keep_classes]  # (n_keep, x, y, z)
                else:
                    data_probabilities = data['probabilities']
                
                print(f"\n Data is of shape {np.shape(data['probabilities'])}\n\n")
                
                # Combine data: concatenate numpy arrays per patient
                for key in data.keys():
                    print(key)
                    if key == 'probabilities':
                        arr = data_probabilities
                    else:
                        arr = data[key]
                    
                    if key in combined_data:
                        combined_data[key] = np.concatenate((combined_data[key], arr[np.newaxis, :,:,:,:]))
                        print(f"Shape of combined data is {np.shape(combined_data['probabilities'])}")
            
                    else:
                        combined_data[key] = arr
                        combined_data[key] = combined_data[key][np.newaxis, :,:,:,:]
#                OLD                       ####                        
#                for key in data.keys():
#                    print(key)
#                    if key in combined_data:
#                        combined_data[key] = np.concatenate((combined_data[key], data[key][np.newaxis, :,:,:,:]))
#                    else:
#                        combined_data[key] = data[key]
#                        combined_data[key] = combined_data[key][np.newaxis, :,:,:,:]
            print(f"Shape of combined data is {np.shape(combined_data['probabilities'])}")
            
            print("Finding keys in combined data")
            for key, value in combined_data.items():
                print(key)
                
            for metric in metrics:
                if metric == "entropy":
                    logging.info(f"Computing an entropy uncertainty map")
                    print(combined_data)
                    uncertainty_map = entropy_prob(combined_data['probabilities'], classes)
                elif metric =="classwise_entropy": 
                    logging.info(f"Computing a classwise entropy uncertainty map")
                    uncertainty_map = entropy_classwise_prob(combined_data['probabilities'], classes)
                elif metric == "mutual_information":
                    logging.info(f"Computing a mutual information uncertainty map")
                    uncertainty_map = mutual_information_prob(combined_data['probabilities'], classes)
                elif metric == "classwise_mutual_information":
                    logging.info(f"Computing a classwise mutual information uncertainty map")
                    uncertainty_map = mutual_information_classwise_prob(combined_data["probabilities"], classes)
                elif metric == "variance":
                    logging.info(f"Computing a variance uncertainty map")
                    uncertainty_map = variance_prob(np.array(combined_data['probabilities']), classes)
                elif metric == "classwise_variance":
                    logging.info(f"computing a classwise variance uncertainty map")
                    uncertainty_map = variance_classwise_prob(np.array(combined_data['probabilities']))
                out_file = os.path.join(output_dir, f"uncertainty_map_{metric}_{patient}.nii.gz")
                save_uncertainty_map_as_nifti(uncertainty_map, out_file, roi_dict=roi_dict_data)
        
            
def main():
    parser = argparse.ArgumentParser(description="Compute uncertainty maps from prediction npz files.")
    parser.add_argument("--folder", type=str, required=True, help="Folder with npz prediction files")
    parser.add_argument("--patients", type=str, nargs="+", required=True, default=None, help="List of patient IDs to process")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save uncertainty maps")
    parser.add_argument("--classes", type=int, required=True, help="Number of predicted classes")
    parser.add_argument("--keep_classes", type=str, nargs="+", default=None, help="List of classes to keep in uncertainty map")
    parser.add_argument("--methods", nargs="+", required=True, help="List of uncertainty methods used to obtain the samples (mc_dropout, deep_ensemble, tta)") 
    parser.add_argument("--metrics", nargs="+", required=True, help="List of uncertainty metrics to compute (entropy, mutual_information, variance, classwise_entorpy)")
    parser.add_argument("--roi_dict", type=str, default=None, help="Path to Python file containing ROI_DICT (e.g., roi_dict.py)")

    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.folder, "uncertainty_maps")
    else:
        args.output_dir = os.path.join(args.output_dir, "uncertainty_maps")
 
    os.makedirs(args.output_dir, exist_ok=True)
    
    create_uncertainty_map_from_samples(folder=args.folder, patients=args.patients, output_dir=args.output_dir, classes=args.classes, keep_classes=args.keep_classes, metrics=args.metrics, methods=args.methods, roi_dict=args.roi_dict)

if __name__ == "__main__":
    main()
