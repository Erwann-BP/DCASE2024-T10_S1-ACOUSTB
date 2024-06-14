# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:28:42 2024

@author: e.betton-ployon
"""

import numpy as np
from scipy.signal import find_peaks, correlate2d
import math

import torch
import yaml

from .models.gcc import GCCPhat


def sigmoid_right(x, edges):
    """
    Returns the value in x of a sigmoid-like function defined between the given edges. 

    Parameters
    ----------
    x : (float) Abscissa value.
    edges : (list of 2 floats) Definition limits of the sigmoid-like function.

    Returns
    -------
    val : (float) Value of the sigmoid-like function in x.

    """
    new_x = (x-np.mean(edges))*2/(max(edges)-np.mean(edges))
    val = 1 / (1 + math.exp(new_x)) * 8 + 4
    return val


def create_right_passby_mask(slope):
    """
    Creates a mask simulating the GCC-Phat of a right pass-by, given a slope parameter.

    Parameters
    ----------
    slope : (float) Coefficient that impacts the slope of the generated mask.

    Returns
    -------
    ref_mask_right : (array of bool) Mask simulating the GCC-Phat of a right pass-by
    

    """
    ref_mask_right = np.zeros((16, int(1.5*2000/slope)), dtype=bool)
    
    edge1 = int(0.25*2000/slope)
    edge2 = int(1.25*2000/slope)
    for i in range(edge1,edge2+1):
        ref_mask_right[int(sigmoid_right(i, [edge1, edge2])):int(sigmoid_right(i, [edge1, edge2]))+2, i] = 1
    
    return ref_mask_right

def get_param_mask(distance, speed_limit, direction):
    """
    Returns the optimized slope-parameter to use to generate a mask.

    Parameters
    ----------
    distance : (float) Distance to street-side of the recording device.
    speed_limit : (float) Speed limit on the studied road. 
    direction : (str) Pass-by direction (left or right)

    Returns
    -------
    optimized_slope : (float) Optimized slope to use to generate a mask.

    """
    
    if direction == "right":
        return -0.63*distance + 0.25*speed_limit+11.89
    
    elif direction == "left":
        return -5.82*distance + 1.01*speed_limit-2.83

def evol_passbys(clean_img, ref):
    """
    Application of the generated mask on a GCC-Phat output, computed on channels 1 and 2 of a real audio record.

    Parameters
    ----------
    clean_img : (2-D array) Output of the GCC-Phat algorithm for channels 1 and 2.
    ref : (2-D array) Generated mask.

    Returns
    -------
    evol: (1-D array) Evolution of the mask-image correlation over the whole GCC-Phat output.

    """
    
    vector = []
    for i in range(1874):
        score = ref[:,0:min(i+len(ref[0]),1873)-i]*clean_img[:,i:min(i+len(ref[0]),1873)]
        vector.append(np.sum(score))
    
    return np.roll(np.array(vector),len(ref[0])//2)


def full_evaluation_corr(audio, site, real_root, n_channels=4, n_gcc=16, stft_params=dict({'hop_length': 512, 'return_complex': True, 'n_fft': 1024, 'center': False})):
    """
    Estimates the counting of vehicles passing by in each direction on the given audio segment.

    Parameters
    ----------
    audio : ((1, n_channels, n_samples) torch.Tensor) Audio content.
    site : (str) Recording site alias.
    real_root : (Path) Path to the real data folder of the given site.
    n_channels : (int, optional) Number of channels in the audio file. The default is 4.
    n_gcc : (int, optional) Number of coefficients to use in the GCC-Phat computation. The default is 16.
    stft_params : (dict, optional) Stft main parameters. The default is dict({'hop_length': 512, 'return_complex': True, 'n_fft': 1024, 'center': False}).

    Returns
    -------
    estim_tuple : (2 int) Estimation of the amount of vehicles passing by in each direction on the given audio segment.

    """
    # Use of the site metadata to estimate the best parameters for mask slope.
    yaml_site_metadata_path = f"{real_root}//meta.json"
    with open(yaml_site_metadata_path, 'r') as file:
        metadata = yaml.safe_load(file)
    
    ref_right = np.flip(create_right_passby_mask(get_param_mask(metadata["geometry"]["distance-to-street-side"], metadata["traffic"]["max-pass-by-speed"], direction="right")),axis=-1)
    ref_left = create_right_passby_mask(get_param_mask(metadata["geometry"]["distance-to-street-side"], metadata["traffic"]["max-pass-by-speed"], direction="left"))           

    yaml_params_path = f".//atsc.//configs//training//{site}//gcc_estimation_params.yaml"
    with open(yaml_params_path, 'r') as file:
        params = yaml.safe_load(file)
    
    params["left"]["width"] = tuple(int(element) for element in params["left"]["width"].split(","))
    params["right"]["width"] = tuple(int(element) for element in params["right"]["width"].split(","))
    
    # Compute GCC-Phat on the chosen audio segment.
    stft_flat = torch.stft(audio, **stft_params, window=torch.hann_window(stft_params["n_fft"]))
    n_time_frames = stft_flat.shape[-1]
    stft = stft_flat.view(1, n_channels, -1, n_time_frames)
    
    # Output: (batch_size, n_channels, n_gcc, n_time_frames) real
    gcc_layer = GCCPhat(n_gcc)
    gcc_norm_layer = torch.nn.BatchNorm1d(num_features=(n_channels * (n_channels - 1)) // 2 * n_gcc)
    gcc_out = gcc_layer(stft)

    # Output: (batch_size, n_channels, n_gcc, n_time_frames) real
    gcc_norm = gcc_norm_layer(gcc_out.reshape(1, -1, n_time_frames)).reshape(*gcc_out.shape)
      
    gcc_norm0 = gcc_norm.detach().numpy()[:,0,:,:].squeeze()
    
    # Correlation between the generated mask and the computed GCC-Phat
    corr_left = correlate2d(gcc_norm0, ref_left)[8:24]
    corr_right = correlate2d(gcc_norm0, ref_right)[8:24]
    evol_left_pb = evol_passbys(corr_left, ref_left)
    evol_right_pb = evol_passbys(corr_right, ref_right)
    
    # Correlation analysis to detect pass-bys in each direction
    count_left, props_left = find_peaks(evol_left_pb, height=params["left"]["height"], prominence=params["left"]["prominence"], distance=params["left"]["distance"], width=params["left"]["width"])
    count_right, props_right = find_peaks(evol_right_pb, height=params["right"]["height"], prominence=params["right"]["prominence"], distance=params["right"]["distance"], width=params["right"]["width"])

    # It seems (after observing audio segments) that loc1 and loc3 left pass-bys correspond to the other sites right pass-bys (maybe related to a different microphone array positioning).
    if site not in ['loc1', 'loc3']:
        return len(count_left), len(count_right)

    else:
        return len(count_right), len(count_left)
