# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:28:42 2024

@author: e.betton-ployon
"""

from scipy import ndimage
from skimage import filters

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, correlate2d
import math

import torch
from torch import Tensor
import yaml

from .models.gcc import GCCPhat


def sigmoid_right(x, edges):
    new_x = (x-np.mean(edges))*2/(max(edges)-np.mean(edges))
    sig = 1 / (1 + math.exp(new_x)) * 8 + 4
    return sig

def create_right_passby_mask(max_speed):
    
    ref_mask_right = np.zeros((16, int(1.5*2000/max_speed)), dtype=bool)
    
    ### La diagonale commence à l'indice int(0.25*2000/max_speed) et se termine à l'indice int(1.25*2000/max_speed)
    # ref_mask_left[11:13, int(0.25*2000/max_speed):int(0.4*2000/max_speed)] = 1
    # ref_mask_left[4:6, int(1.1*2000/max_speed)-1:int(1.25*2000/max_speed)-1] = 1
    
    edge1 = int(0.25*2000/max_speed)
    edge2 = int(1.25*2000/max_speed)
    for i in range(edge1,edge2+1):
        # print(int(sigmoid_left((int(0.35*2000/max_speed)-i)))-1)
        # print(int(sigmoid_left((int(0.35*2000/max_speed)-i)))+1)
        # print(int(0.4*2000/max_speed)+i)
        # print("\n\n")
        ref_mask_right[int(sigmoid_right(i, [edge1, edge2])):int(sigmoid_right(i, [edge1, edge2]))+2, i] = 1
        # ref_mask_left[int(4+i*8/dist):int(6+i*8/dist), int(1.1*2000/max_speed)-i] = 1
    
    return ref_mask_right


# def create_batch(audio_filepath):
#     data, sr = sf.read(audio_filepath)
#     batch = {'audio':torch.unsqueeze(Tensor(data),0), 
#              'wn':Tensor([0]),
#              'dow':Tensor([0]),
#              'hour':Tensor([17]),
#              'minute':Tensor([0]),
#              'vehic_left':Tensor([21]),
#              'vehic_right':Tensor([6]),
#              'batch_idx':Tensor([0])}
#     return batch


# def evaluate_batch(model, batch, audio):
#     batch_size, n_channels, _ = audio.shape

#     # Output: (batch_size, n_channels, n_fft/2+1, n_time_frames) complex
#     audio_flat = audio.view(batch_size * n_channels, -1)
#     stft_flat = torch.stft(audio_flat, **model._stft_params, window=model._stft_window)
#     n_time_frames = stft_flat.shape[-1]
#     stft = stft_flat.view(batch_size, n_channels, -1, n_time_frames)
#     # print(stft)

#     # Output: (batch_size, n_channels, n_gcc, n_time_frames) real
#     gcc_out = model.gcc(stft)

#     # Output: (batch_size, n_channels, n_gcc, n_time_frames) real
#     gcc_norm = model.gcc_norm(gcc_out.reshape(batch_size, -1, n_time_frames)).reshape(*gcc_out.shape)
    
#     return gcc_norm.detach().numpy()[:,0,:,:].squeeze()

def get_param_mask(distance, speed_limit, direction, site):
    
    if direction == "right":
        return -0.63*distance + 0.25*speed_limit+11.89
    
    elif direction == "left":
        return -5.82*distance + 1.01*speed_limit-2.83


def get_binary(gcc_norm_numpy):
    binary = (256*(gcc_norm_numpy > 1)).astype(np.float64)
    binary[0:4,:] = 0
    binary[13:,:] = 0
    return binary

def clean(binary, mask_diag):
    
    struct = np.array([[False, False, False], [ True,  True, False], [False, False, False]])
    
    test = filters.correlate_sparse(binary, mask_diag).astype(binary.dtype)
    test2 = filters.correlate_sparse(test>400, mask_diag).astype(binary.dtype)
    test3 = ndimage.binary_erosion(test2>1.5, structure=struct).astype(binary.dtype)
    
    # test4 = filters.correlate_sparse(test3, struct_5).astype(binary.dtype)
    
    return test3

def evol_passbys(clean_img, ref):
    
    vector = []
    for i in range(1874):
        score = ref[:,0:min(i+len(ref[0]),1873)-i]*clean_img[:,i:min(i+len(ref[0]),1873)]
        vector.append(np.sum(score))
    
    return np.roll(np.array(vector),len(ref[0])//2)

def plot_passby_detection_arrays(clean_img, vector, direction="left"):
    
    fig, ax = plt.subplots(2,1)
    ax[0].set_title(f"Sortie du GCC-norm traitée pour faire apparaître les passages '{direction}'")
    ax[0].imshow(clean_img, aspect='auto', cmap='gray')
    ax[1].set_title(f"Application du masque 'ref_{direction}' sur l'image ci-dessus pour détecter les passages dans le sens '{direction}'")
    ax[1].plot(vector)
    ax[1].margins(x=0)
    fig.subplots_adjust(hspace=0.5)


def full_evaluation_corr(audio, site, real_root, n_channels=4, n_gcc=16, stft_params=dict({'hop_length': 512, 'return_complex': True, 'n_fft': 1024, 'center': False})):
       
    ### Construire les 4 variables suivantes en fonction du site et des propriétés enregistrées
    yaml_site_metadata_path = f"{real_root}//meta.json"
    with open(yaml_site_metadata_path, 'r') as file:
        metadata = yaml.safe_load(file)
    
    ref_right = np.flip(create_right_passby_mask(get_param_mask(metadata["geometry"]["distance-to-street-side"], metadata["traffic"]["max-pass-by-speed"], direction="right", site=site)),axis=-1)
    ref_left = create_right_passby_mask(get_param_mask(metadata["geometry"]["distance-to-street-side"], metadata["traffic"]["max-pass-by-speed"], direction="left", site=site))           

    yaml_params_path = f".//atsc.//configs//training//{site}//gcc_estimation_params.yaml"
    with open(yaml_params_path, 'r') as file:
        params = yaml.safe_load(file)
    
    params["left"]["width"] = tuple(int(element) for element in params["left"]["width"].split(","))
    params["right"]["width"] = tuple(int(element) for element in params["right"]["width"].split(","))
    
    # audio_flat = audio.view(n_channels, -1)
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
    
    # binary_img = get_binary(gcc_norm0)
    
    corr_left = correlate2d(gcc_norm0, ref_left)[8:24]
    corr_right = correlate2d(gcc_norm0, ref_right)[8:24]
    
    evol_left_pb = evol_passbys(corr_left, ref_left)
    evol_right_pb = evol_passbys(corr_right, ref_right)
    
    count_left, props_left = find_peaks(evol_left_pb, height=params["left"]["height"], prominence=params["left"]["prominence"], distance=params["left"]["distance"], width=params["left"]["width"])
    count_right, props_right = find_peaks(evol_right_pb, height=params["right"]["height"], prominence=params["right"]["prominence"], distance=params["right"]["distance"], width=params["right"]["width"])

    # return len(count_left), len(count_right)
    
    if site not in ['loc1', 'loc3']:
        return len(count_left), len(count_right)

    else:
        return len(count_right), len(count_left)
