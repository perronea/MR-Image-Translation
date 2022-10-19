#!/bin/sh

t1w=$1
atl_t1w=$2
atl_t2w=$3
export ANTSPATH=$4

# short versions of names
short_t1w=`echo $t1w | rev | cut -d "/" -f1 | rev | cut -d "." -f1`
short_atl_t1w=`echo $atl_t1w | rev | cut -d "/" -f1 | rev | cut -d "." -f1`
short_atl_t2w=`echo $atl_t2w | rev | cut -d "/" -f1 | rev | cut -d "." -f1`
fin_t1=`echo $short_atl_t1w | sed 's/_init/_final/'`
fin_t2=`echo $short_atl_t2w | sed 's/_init/_final/'`
threads=`nproc`

# run ANTs registration
${ANTSPATH}/antsRegistrationSyN.sh -d 3 -f ${atl_t1w} -m ${t1w} -o ${short_t1w}_to_${short_atl_t1w} -p f -n ${threads}

# apply warps to T1
${ANTSPATH}/antsApplyTransforms -d 3 --i ${atl_t1w} -n BSpline -t [${short_t1w}_to_${short_atl_t1w}0GenericAffine.mat,1] -t ${short_t1w}_to_${short_atl_t1w}1InverseWarp.nii.gz -o ${fin_t1}.nii.gz -r ${t1w}

# apply warps to T2
${ANTSPATH}/antsApplyTransforms -d 3 --i ${atl_t2w} -n BSpline -t [${short_t1w}_to_${short_atl_t1w}0GenericAffine.mat,1] -t ${short_t1w}_to_${short_atl_t1w}1InverseWarp.nii.gz -o ${fin_t2}.nii.gz -r ${t1w}


