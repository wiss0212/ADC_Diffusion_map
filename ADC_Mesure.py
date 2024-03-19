from pathlib import Path # pathlib for easy path handling
import pydicom # pydicom to handle dicom files
import matplotlib.pyplot as plt
import numpy as np
import dicom2nifti # to convert DICOM files to the NIftI format
import nibabel as nib # nibabel to handle nifti files
import matplotlib.pyplot as plt
import os
import argparse
import logging
from skimage.measure import label
from scipy.ndimage import binary_fill_holes, binary_dilation,\
    binary_erosion
from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data
from scipy.ndimage import zoom
from medpy.io import load, save, header
from medpy.filter import otsu



def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    largest=max(list_seg, key=lambda x:x[1])[0]
    labels_max=(labels == largest).astype(int)
    return labels_max

    
head_mri_dicom_b0 = Path(" use your path ")
head_mri_dicom_b1000 = Path(" use your path ")
dicom2nifti.convert_directory(head_mri_dicom_b0, ".")
dicom2nifti.convert_directory(head_mri_dicom_b1000, ".")
nifti_b0=nib.load("load your b0 volume ")
nifti_b1000=nib.load("load the other b value  ")
print(nifti_b0)
   # convert to float
data_b0= nifti_b0.get_fdata().astype(float)
data_b1000 =nifti_b1000.get_fdata().astype(float)

data_b0_abs=np.abs(data_b0)
data_b1000_abs=np.abs(data_b1000)
print(data_b0_abs.shape)
print(data_b1000_abs.shape)
print(data_b1000)
#resample data_b0
resampled_b0 = zoom(data_b0_abs, (0.3636, 0.3636, 1.125, 1), order=1)
result_data_b0=resampled_b0[:,:,:,0]
print(result_data_b0.shape)
 # compute threshold value
b0thr = otsu(result_data_b0, 32) / 4.  # divide by 4 to decrease impact
bxthr = otsu(data_b1000_abs, 32) / 4.
mask = binary_fill_holes(result_data_b0 > b0thr) & binary_fill_holes(data_b1000_abs > bxthr)

    #  binary morphology steps 
mask = binary_erosion(mask, iterations=1)
mask = getLargestCC(mask)
mask = binary_dilation(mask, iterations=1)

# ADC  calcul without Mask approche 
ratio=np.divide(data_b1000_abs,result_data_b0, where=result_data_b0 != 0)
print(ratio)
result_ADC = -1000 * np.log(ratio)

# ADC calcul with mask 
# compute the ADC
adc = np.zeros(result_data_b0.shape, result_data_b0.dtype)
adc[mask] = -1.* 1000 * np.log(data_b1000_abs[mask] / result_data_b0[mask])
adc[adc < 0] = 0

#save adc 
nifti_result = nib.Nifti1Image(adc, affine=None)
output_path="C:\\Users\daouiaouissem\\Downloads\\Clef\\ResultatsCodePython.nii.gz"
save(adc, output_path)

#Display a slice of the calculated ADC using the MASK approach
z = 17
plt.imshow(adc[:, :, z].T, origin='lower', cmap='gray',
           interpolation='nearest')
plt.axhline(y=100)
plt.axvline(x=170)
plt.savefig("data_slice.png")

#Display a slice of the calculated ADC without Mask approch 
z = 17
plt.imshow(result_ADC[:, :, z].T, origin='lower', cmap='gray',
           interpolation='nearest')
plt.axhline(y=100)
plt.axvline(x=170)
plt.savefig("data_slice.png")
