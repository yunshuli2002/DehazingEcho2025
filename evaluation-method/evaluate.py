from pathlib import Path
import fid_score
from glob import glob
import torch
import numpy as np 
import cv2
from scipy.stats import ks_2samp  

def calculate_fid_score(denoised_image_dirs):
    if isinstance(denoised_image_dirs, (str, Path)):
        denoised_image_dirs = [denoised_image_dirs]
    elif not isinstance(denoised_image_dirs, list):
        raise ValueError("Input must be a path or list of paths")

    ground_truth_dir = Path("/opt/ml/input/data/ground_truth")
    clean_images_folder = glob(str(ground_truth_dir) + "/clean" + '/*.png')

    fid_value = fid_score.calculate_fid_given_paths(
        [clean_images_folder, denoised_image_dirs],
        batch_size=32,
        num_workers=0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048
    )
    return fid_value

def gcnr(img1, img2):
    """Generalized Contrast-to-Noise Ratio"""
    _, bins = np.histogram(np.concatenate((img1, img2)), bins=256)
    f, _ = np.histogram(img1, bins=bins, density=True)
    g, _ = np.histogram(img2, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))

def cnr(img1, img2):
    """Contrast-to-Noise Ratio"""
    return (img1.mean() - img2.mean()) / np.sqrt(img1.var() + img2.var())

def calculate_cnr_gcnr(result_dehazed_cardiac_ultrasound, mask_path):
    """
    Evaluate gCNR and CNR metrics for denoised images using paired masks.
    Saves detailed and summary statistics to Excel.
    """
    results = []

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    roi1_pixels = result_dehazed_cardiac_ultrasound[mask == 255]  # Foreground ROI
    roi2_pixels = result_dehazed_cardiac_ultrasound[mask == 128]  # Background/Noise ROI
    
    gcnr_val = gcnr(roi1_pixels, roi2_pixels)
    cnr_val = cnr(roi1_pixels, roi2_pixels)
   
    results.append([cnr_val, gcnr_val])

    return results

def calculate_ks_statistics(result_hazy_cardiac_ultrasound, result_dehazed_cardiac_ultrasound, mask_path):
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    roi1_original = result_hazy_cardiac_ultrasound[mask == 255]  # region A
    roi1_denoised = result_dehazed_cardiac_ultrasound[mask == 255]
    roi2_original = result_hazy_cardiac_ultrasound[mask == 128]  # region B
    roi2_denoised = result_dehazed_cardiac_ultrasound[mask == 128]

    roi1_ks_stat, roi1_ks_p_value = (None, None)
    roi2_ks_stat, roi2_ks_p_value = (None, None)

    if roi1_original.size > 0 and roi1_denoised.size > 0:
        roi1_ks_stat, roi1_ks_p_value = ks_2samp(roi1_original, roi1_denoised)

    if roi2_original.size > 0 and roi2_denoised.size > 0:
        roi2_ks_stat, roi2_ks_p_value = ks_2samp(roi2_original, roi2_denoised)

    return roi1_ks_stat, roi1_ks_p_value, roi2_ks_stat, roi2_ks_p_value

from scipy.ndimage import binary_erosion
from test import inference # Our Segmentation Method
from PIL import Image
from scipy.ndimage import distance_transform_edt 

def calculate_dice_asd(image_path, label_path, checkpoint_path, image_size=224):
   
    pred_img = inference(image_path, checkpoint_path, image_size)
    pred = np.array(pred_img) > 127  

    label = Image.open(label_path).convert('L')
    label = label.resize((image_size, image_size), Image.NEAREST)
    label = np.array(label) > 127  

    # calculate Dice
    intersection = np.logical_and(pred, label).sum()
    dice = 2 * intersection / (pred.sum() + label.sum() + 1e-8)

    # calculate ASD
    if pred.sum() == 0 or label.sum() == 0:
        asd = np.nan  
    else:
        pred_dt = distance_transform_edt(~pred)
        label_dt = distance_transform_edt(~label)

        surface_pred = pred ^ binary_erosion(pred)
        surface_label = label ^ binary_erosion(label)

        d1 = pred_dt[surface_label].mean()
        d2 = label_dt[surface_pred].mean()
        asd = (d1 + d2) / 2

    return dice, asd

def calculate_final_score(aggregates):
    try:
        # (FID + CNR + gCNR):(KS^A + KS^B):(Dice + ASD)= 5:3:2
        
        group1_score = 0  # FID + CNR + gCNR
        if aggregates.get("fid") is not None:
            fid_min = 60.0
            fid_max = 150.0
            fid_score = (fid_max -  aggregates["fid"])/(fid_max - fid_min)
            fid_score = max(0, min(1, fid_score))
            group1_score += fid_score * 100 * 0.33
        
        if aggregates.get("cnr_mean") is not None:
            cnr_min = 1.0
            cnr_max = 1.5
            cnr_score = (aggregates["cnr_mean"] - cnr_min)/(cnr_max - cnr_min)
            cnr_score = max(0, min(1, cnr_score))
            group1_score += cnr_score * 100 * 0.33
        
        if aggregates.get("gcnr_mean") is not None:
            gcnr_min = 0.5
            gcnr_max = 0.8
            gcnr_score = (aggregates["gcnr_mean"] - gcnr_min)/(gcnr_max - gcnr_min)
            gcnr_score = max(0, min(1, gcnr_score))
            group1_score += gcnr_score * 100 * 0.34
            
        group2_score = 0  # KS^A + KS^B
        if aggregates.get("ks_roi1_ksstatistic_mean") is not None:
            ks1_min = 0.1
            ks1_max = 0.3
            ks1_score = (ks1_max - aggregates["ks_roi1_ksstatistic_mean"])/(ks1_max - ks1_min)
            ks1_score = max(0, min(1, ks1_score))
            group2_score += ks1_score * 100 * 0.5
        
        if aggregates.get("ks_roi2_ksstatistic_mean") is not None:
            ks2_min = 0.0
            ks2_max = 0.5
            ks2_score = (aggregates["ks_roi2_ksstatistic_mean"] - ks2_min)/(ks2_max - ks2_min)
            ks2_score = max(0, min(1, ks2_score))
            group2_score += ks2_score * 100 * 0.5
            
        group3_score = 0  # Dice + ASD
        if aggregates.get("dice_mean") is not None:
            dice_min = 0.85
            dice_max = 0.95
            dice_score = (aggregates["dice_mean"] - dice_min)/(dice_max - dice_min)
            dice_score = max(0, min(1, dice_score))
            group3_score += dice_score * 100 * 0.5
        if aggregates.get("asd_mean") is not None:
            asd_min = 0.7
            asd_max = 2.0
            asd_score = (asd_max - aggregates["asd_mean"])/(asd_max - asd_min)
            asd_score = max(0, min(1, asd_score))
            group3_score += asd_score * 100 * 0.5
            
        # Final score calculation
        final_score = (group1_score * 5 + group2_score * 3 + group3_score * 2 ) / 10
        
        return final_score
        
    except Exception as e:
        print(f"Error calculating final score: {str(e)}")
        return 0


