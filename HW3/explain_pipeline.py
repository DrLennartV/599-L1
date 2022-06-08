import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import os
import time
from tqdm import tqdm, trange
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import _CNN
from data_util import CNN_Data
import data_util
import pandas as pd


# This is a color map that you can use to plot the SHAP heatmap on the input MRI
colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


# Returns two data loaders (objects of the class: torch.utils.data.DataLoader) that are
# used to load the background and test datasets.
def prepare_dataloaders(bg_csv, test_csv, bg_batch_size = 8, test_batch_size= 1, num_workers=1):
    '''
    Attributes:
        bg_csv (str): The path to the background CSV file.
        test_csv (str): The path to the test data CSV file.
        bg_batch_size (int): The batch size of the background data loader
        test_batch_size (int): The batch size of the test data loader
        num_workers (int): The number of sub-processes to use for dataloader
    '''
    # YOUR CODE HERE
    bgset = CNN_Data(bg_csv)
    bgloader = DataLoader(bgset, bg_batch_size, num_workers)
    testset = CNN_Data(test_csv)
    testloader = DataLoader(testset, test_batch_size, num_workers)
    return bgloader, testloader

# Generates SHAP values for all pixels in the MRIs given by the test_loader
def create_SHAP_values(bg_loader, test_loader, mri_count, save_path):
    '''
    Attributes:
        bg_loader (torch.utils.data.DataLoader): Dataloader instance for the background dataset.
        test_loader (torch.utils.data.DataLoader): Dataloader instance for the test dataset.
        mri_count (int): The total number of explanations to generate.
        save_path (str): The path to save the generated SHAP values (as .npy files).
    '''
    # YOUR CODE HERE
    # randomize a batch
    bg_batch = next(iter(bg_loader))
    bg_imgs, bg_names, _ = bg_batch
    background = bg_imgs
    test_it = iter(test_loader)
    for i in range(mri_count):
        test_batch = next(test_it)
        test_imgs, test_names, _ = test_batch
        e = shap.DeepExplainer(myModel, background)
        shap_values = e.shap_values(test_imgs)  # the calculated shap value
        print(test_names[0])
        np.save(f'{save_path}/{test_names[0]}', np.array(shap_values))
    pass

# Aggregates SHAP values per brain region and returns a dictionary that maps
# each region to the average SHAP value of its pixels.
def aggregate_SHAP_values_per_region(shap_values, seg_path, brain_regions):
    '''
    Attributes:
        shap_values (ndarray): The shap values for an MRI (.npy).
        seg_path (str): The path to the segmented MRI (.nii).
        brain_regions (dict): The regions inside the segmented MRI image (see data_utl.py)
    '''
    # YOUR CODE HERE
    shap_val = np.load(shap_values_path)
    img = nib.load(seg_path).dataobj
    I, J, K = np.shape(img)

    pos_shap_dc = {}
    neg_shap_dc = {}
    num_dc = {}

    for k in brain_regions.keys():
        pos_shap_dc[k] = 0
        neg_shap_dc[k] = 0
        num_dc[k] = 0

    for i in trange(I):
        for j in range(J):
            for k in range(K):
                if img[i, j, k] != 0.0:
                    pos_shap_dc[img[i, j, k]] += shap_val[0, 0, 0, i, j, k]
                    neg_shap_dc[img[i, j, k]] += shap_val[1, 0, 0, i, j, k]
                    num_dc[img[i, j, k]] += 1
    pos_avg_shap_dc = {}
    neg_avg_shap_dc = {}
    for k in num_dc.keys():
        pos_avg_shap_dc[brain_regions[k]] = pos_shap_dc[k] / num_dc[k]
        neg_avg_shap_dc[brain_regions[k]] = neg_shap_dc[k] / num_dc[k]
    return pos_avg_shap_dc, neg_avg_shap_dc

def calc_alltest_avg_SHAP():
    file_ls = pd.read_csv(r'ADNI3/test_data.csv')['filename']
    brain_regions = data_util.brain_regions
    num = len(file_ls)
    file_prefix_ls = []
    for file in file_ls:
        file_prefix_ls.append(file.split('/')[-1].split('.')[0])
    shap_val_path_ls = []
    seg_path_ls = []
    for prefix in file_prefix_ls:
        shap_val_path_ls.append(f'ADNI3/shape_values/{prefix}.npy')
        seg_path_ls.append(f'ADNI3/seg/{prefix}.nii')
    pos_avg_shap_dc = {}
    neg_avg_shap_dc = {}
    for k in brain_regions.values():
        pos_avg_shap_dc[k] = 0
        neg_avg_shap_dc[k] = 0
    for shap_val_path, seg_path in zip(shap_val_path_ls, seg_path_ls):
        pos_dc, neg_dc = aggregate_SHAP_values_per_region(shap_val_path, seg_path, brain_regions)
        for k in brain_regions.values():
            pos_avg_shap_dc[k] += pos_dc[k]
            neg_avg_shap_dc[k] += neg_dc[k]
    for k in brain_regions.values():
        pos_avg_shap_dc[k] /= num
        neg_avg_shap_dc[k] /= num
    return pos_avg_shap_dc, neg_avg_shap_dc

# Returns a list containing the top-10 most contributing brain regions to each predicted class (AD/NotAD).
def output_top_10_lst(csv_file):
    '''
    Attribute:
        csv_file (str): The path to a CSV file that contains the aggregated SHAP values per region.
    '''
    # YOUR CODE HERE
    pos_avg_shap_dc, neg_avg_shap_dc = calc_alltest_avg_SHAP()
    pos_avg_shap_dc = sorted(pos_avg_shap_dc.items(), key=lambda x: x[1], reverse=True)
    neg_avg_shap_dc = sorted(neg_avg_shap_dc.items(), key=lambda x: x[1], reverse=True)

    pos_name_ls = []
    neg_name_ls = []
    for p, n in zip(pos_avg_shap_dc, neg_avg_shap_dc):
        pos_name_ls.append(p[0])
        neg_name_ls.append(n[0])
    df = pd.DataFrame(data={'Positive': pos_name_ls, 'Negative': neg_name_ls})
    df.to_csv(csv_file, index=True)
    pass

# Plots SHAP values on a 2D slice of the 3D MRI.
def plot_shap_on_mri(subject_mri, shap_values):
    '''
    Attributes:
        subject_mri (str): The path to the MRI (.npy).
        shap_values (str): The path to the SHAP explanation that corresponds to the MRI (.npy).
    '''
    # YOUR CODE HERE
    shap_values = np.load(shap_values_path)
    subject_mri = np.load(subject_mri_path)
    subject_mri = subject_mri[np.newaxis, np.newaxis, :, :, :]
    shap_values = np.swapaxes(shap_values, -4, -1)
    subject_mri = np.swapaxes(subject_mri, -4, -1)

    shap_slice = np.mean(shap_values, 4, keepdims=False)
    mri_slice = np.mean(subject_mri, 3, keepdims=False)
    shap_slice = [s for s in shap_slice]
    shap.image_plot(shap_slice, -mri_slice, show=False)
    plt.savefig(f'ADNI3/heatmaps/{heatmap_name}_1.png')

    shap_slice = np.mean(shap_values, 3, keepdims=False)
    mri_slice = np.mean(subject_mri, 2, keepdims=False)
    shap_slice = [s for s in shap_slice]
    shap.image_plot(shap_slice, -mri_slice, show=False)
    plt.savefig(f'ADNI3/heatmaps/{heatmap_name}_2.png')

    shap_slice = np.mean(shap_values, 2, keepdims=False)
    mri_slice = np.mean(subject_mri, 1, keepdims=False)
    shap_slice = [s for s in shap_slice]
    shap.image_plot(shap_slice, -mri_slice, show=False)
    plt.savefig(f'ADNI3/heatmaps/{heatmap_name}_3.png')


if __name__ == '__main__':
global myModel
    myModel = _CNN(20, 0.1)
    myModel_dict = torch.load("ADNI3/model_para/cnn_best.pth", map_location=torch.device('cpu'))
    myModel.load_state_dict(myModel_dict["state_dict"])

    # TASK I: Load CNN model and isntances (MRIs)
    #         Report how many of the 19 MRIs are classified correctly
    # YOUR CODE HERE

	bg_loader, test_loader = prepare_dataloaders('ADNI3/bg_data.csv', 'ADNI3/test_data.csv')
    correct = 0
    name_ls = []
    pre_ls = []
    act_ls = []
    for idx, (mri, name, ad) in enumerate(test_loader, 1):
        pre_ad = myModel(mri)
        pre_ad = pre_ad.max(dim=1, keepdim=False)[1]
        correct += pre_ad.eq(ad).sum().item()
        name_ls.append(name[0])
        pre_ls.append(pre_ad.item())
        act_ls.append(ad.item())
    df = pd.DataFrame(data={'filename': name_ls, 'predict': pre_ls, 'acture': act_ls})
    df.to_csv("ADNI3/Forecast.csv", index=False)
    print(f'correctly predicated:{correct}')

    # TASK II: Probe the CNN model to generate predictions and compute the SHAP
    #          values for each MRI using the DeepExplainer or the GradientExplainer.
    #          Save the generated SHAP values that correspond to instances with a
    #          correct prediction into output/SHAP/data/
    # YOUR CODE HERE

	create_SHAP_values(bg_loader, test_loader, 6, "ADNI3/shape_values")

    # TASK III: Plot an explanation (pixel-based SHAP heatmaps) for a random MRI.
    #           Save heatmaps into output/SHAP/heatmaps/
    # YOUR CODE HERE

    plot_shap_on_mri(
        'ADNI3/data/ADNI_135_S_6545_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190913133638192_107_S873086_I1226543.npy',
        'ADNI3/shape_values/ADNI_135_S_6545_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190913133638192_107_S873086_I1226543.npy',
        'exist_AD'
    )

    plot_shap_on_mri(
        'ADNI3/data/ADNI_099_S_6038_MR_Accelerated_Sag_IR-FSPGR___br_raw_20181112140012329_32_S746377_I1071981.npy',
        'ADNI3/shape_values/ADNI_099_S_6038_MR_Accelerated_Sag_IR-FSPGR___br_raw_20181112140012329_32_S746377_I1071981.npy',
        'non_exist_AD'
    )

    # TASK IV: Map each SHAP value to its brain region and aggregate SHAP values per region.
    #          Report the top-10 most contributing regions per class (AD/NC) as top10_{class}.csv
    #          Save CSV files into output/top10/
    # YOUR CODE HERE
	output_top_10_lst(r'ADNI3/top_10.csv')

    pass
