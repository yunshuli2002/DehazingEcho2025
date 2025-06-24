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


def calculate_final_score(aggregates):

    '''
    The final score calculation method is for reference only. 
    The final FID normalization will be based on the maximum value among the submitted results. 
    For the Dice + ASD values, we will use those measured by our own segmentation model.
    '''
    try:
        # (FID + CNR + gCNR):(KS^A + KS^B):(Dice + ASD)= 5:3:2

        group1_score = 0  # FID + CNR + gCNR
        if aggregates.get("fid") is not None:
            group1_score += (100 -  aggregates["fid"]) * 0.33
        
        if aggregates.get("cnr_mean") is not None:
            group1_score += aggregates["cnr_mean"] * 100 * 0.33
        
        if aggregates.get("gcnr_mean") is not None:
            group1_score += aggregates["gcnr_mean"] * 100 * 0.33
            
        group2_score = 0  # KS^A + KS^B
        if aggregates.get("ks_a_mean") is not None:
            group2_score += (1 - aggregates["ks_a_mean"]) * 100 * 0.5
        
        if aggregates.get("ks_b_mean") is not None:
            group2_score += aggregates["ks_b_mean"] * 100 * 0.5
            
        group3_score = 0  # Dice + ASD
        if aggregates.get("dice_mean") is not None:
            group3_score += aggregates["dice_mean"] * 100 * 0.5
        if aggregates.get("asd_mean") is not None:
            group3_score += (1 - aggregates["asd_mean"]) * 100 * 0.5

        final_score = (group1_score * 5 + group2_score * 3 + group3_score * 2 ) / 10
        
        return final_score
        
    except Exception as e:
        print(f"Error calculating final score: {str(e)}")
        return 0

