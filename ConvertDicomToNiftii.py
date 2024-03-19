from pathlib import Path # pathlib for easy path handling
import pydicom # pydicom to handle dicom files
import matplotlib.pyplot as plt
import numpy as np
import dicom2nifti # to convert DICOM files to the NIftI format
import nibabel as nib # nibabel to handle nifti files
import matplotlib.pyplot as plt
import os
from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data




    
head_mri_dicom_b0 = Path("C:/Users/daouiaouissem/Downloads/Clef/S72660/S6010")
head_mri_dicom_b1000 = Path("C:/Users/daouiaouissem/Downloads/Clef/S72660/S6030")
dicom2nifti.convert_directory(head_mri_dicom_b0, ".")
dicom2nifti.convert_directory(head_mri_dicom_b1000, ".")


nifti=nib.load("C:/Users/daouiaouissem/OneDrive - Median Technologies SA/601_diff_v100.nii.gz")
print(nifti)
#accéder à la valeur associée à la clé "qoffset_x"
nifti.header["qoffset_x"]
print(nifti.shape) # get the image shape
print(nifti.header.get_data_shape()) # get the image shape in a different way
bvals, bvecs = extract_bvalues_and_bvectors_from_nifti("C:/Users/daouiaouissem/OneDrive - Median Technologies SA/601_diff_v100.nii.gz")
print(bvals)
print(bvecs)
# Accéder aux données de volume
data = nifti.get_fdata()
z = 15
b = 1

plt.imshow(data[:, :, z, b].T, origin='lower', cmap='gray',
           interpolation='nearest')
plt.axhline(y=100)
plt.axvline(x=170)
plt.savefig("data_slice.png")
