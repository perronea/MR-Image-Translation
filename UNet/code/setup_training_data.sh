#!/bin/bash

module load fsl
FSLDIR=/panfs/roc/msisoft/fsl/6.0.2
module load ants
module load python3/3.8.3_anaconda2020.07_mamba 

input=$1
WD=$2
output=$3

mkdir -p ${WD}

std=$WD/`basename $input | sed 's|.nii.gz|_std.nii.gz|'`
acpc=$WD/`basename $input | sed 's|.nii.gz|_acpc.nii.gz|'`
omat=`echo $acpc | sed 's|.nii.gz|.mat|'`
denoised=`echo $acpc | sed 's|.nii.gz|Denoise.nii.gz|'`
dc=`echo $denoised | sed 's|.nii.gz|N4dc.nii.gz|'`


fslreorient2std ${input} ${std}

if [[ $input == *T1w* ]]; then
    /home/faird/shared/projects/3D_MRI_GAN/tio_unet/dcan-infant-pipeline/PreFreeSurfer/scripts/ACPCAlignment_with_crop.sh --workindir=${WD} --in=${std} --ref=/home/faird/shared/projects/3D_MRI_GAN/tio_unet/dcan-infant-pipeline/global/templates/INFANT_MNI_T1_1mm.nii.gz --out=$acpc --omat=$omat --brainsize=80
    
    DenoiseImage -d 3 -n Rician --input-image ${acpc} --output ${denoised}

    #N4BiasFieldCorrection -d 3 --input-image ${denoised} --shrink-factor 2 --output [${dc},${WD}/N4BiasField_T1.nii.gz]

    mkdir -p `dirname ${output}`

    cp ${denoised} ${output}/${subject}_${session}_T1w.niigz

else
    /home/faird/shared/projects/3D_MRI_GAN/tio_unet/dcan-infant-pipeline/PreFreeSurfer/scripts/ACPCAlignment_with_crop.sh --workindir=${WD} --in=${std} --ref=/home/faird/shared/projects/3D_MRI_GAN/tio_unet/dcan-infant-pipeline/global/templates/INFANT_MNI_T2_1mm.nii.gz --out=$acpc --omat=$omat --brainsize=80

    DenoiseImage -d 3 -n Rician --input-image ${acpc} --output ${denoised}

    #N4BiasFieldCorrection -d 3 --input-image ${denoised} --shrink-factor 2 --output [${dc},${WD}/N4BiasField_T1.nii.gz]

    mkdir -p `dirname ${output}`

    cp ${denoised} ${output}/${subject}_${session}_T2w.niigz
fi



