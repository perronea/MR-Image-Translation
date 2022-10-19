#!/bin/sh 

##################
### get inputs ###
################## 
t1w=$1
atl_t1w=$2
atl_t2w=$3
ANTSPATH=$4

#########################
### set up parameters ###
#########################
ANTS=${ANTSPATH}/antsRegistration
WARP=${ANTSPATH}/antsApplyTransforms
N4=${ANTSPATH}/N4BiasFieldCorrection
PEXEC=${ANTSPATH}/ANTSpexec.sh
SGE=${ANTSPATH}/waitForSGEQJobs.pl
PBS=${ANTSPATH}/waitForPBSQJobs.pl
XGRID=${ANTSPATH}/waitForXGridJobs.pl
SLURM=${ANTSPATH}/waitForSlurmJobs.pl
DIM=3
USEFLOAT=1
short_t1w=`echo $t1w | rev | cut -d "/" -f1 | rev | cut -d "." -f1`
short_atl_t1w=`echo $atl_t1w | rev | cut -d "/" -f1 | rev | cut -d "." -f1`
short_atl_t2w=`echo $atl_t2w | rev | cut -d "/" -f1 | rev | cut -d "." -f1`

###########################################
### compute initial transforms and warps ##
###########################################
echo "Calculating warp fields for image ${t1w}" > ${short_t1w}.log
cmd0="${ANTS} -d 3 --float 1 --verbose 1 -u 1 -w [ 0.01,0.99 ] -z 1 -r [ ${atl_t1w},${t1w},1 ] "
cmd1="-t Rigid[ 0.1 ] -m MI[ ${atl_t1w},${t1w},0.5,32,Regular,0.25 ] -c [ 1000x500x250x0,1e-6,10 ] -f 6x4x2x1 -s 3x2x1x0 "
cmd2="-t Affine[ 0.1 ] -m MI[ ${atl_t1w},${t1w},0.5,32,Regular,0.25 ] -c [ 1000x500x250x0,1e-6,10 ] -f 6x4x2x1 -s 3x2x1x0 "
cmd3="-t SyN[ 0.1,3,0 ] -m MI[ ${atl_t1w},${t1w},0.5,32,Regular,0.25 ] -c [ 1000x500x250x0,1e-6,10 ] -f 6x4x2x1 -s 3x2x1x0 -o ${short_t1w}_rgd_affn_syn_on_${short_atl_t1w}_"

CMD="${cmd0} ${cmd1} ${cmd2} ${cmd3}"
echo ${CMD} >> ${short_t1w}.log
${CMD} >> ${short_t1w}.log

##############################
### apply transforms to T1 ###
##############################
cmd4="${ANTSPATH}/antsApplyTransforms -d 3 --float 1 --verbose 1 "
cmd5=" -r ${t1w} -o ${short_atl_t1w}_on_${short_t1w}_init.nii.gz "
cmd6="-t ${short_t1w}_rgd_affn_syn_on_${short_atl_t1w}_1InverseWarp.nii.gz -t [${short_t1w}_rgd_affn_syn_on_${short_atl_t1w}_0GenericAffine.mat,1]"
cmd7="-i ${atl_t1w}"

CMD2="${cmd4} ${cmd5} ${cmd6} ${cmd7}"
echo "" >> ${short_t1w}.log
echo "" >> ${short_t1w}.log
echo "Applying warp fields to images ${t1w}" >> ${short_t1w}.log
${CMD2} >> ${short_t1w}.log

##############################
### apply transforms to T2 ###
##############################
cmd8="${ANTSPATH}/antsApplyTransforms -d 3 --float 1 --verbose 1 "
cmd9=" -r ${t1w} -o ${short_atl_t2w}_on_${short_t1w}_init.nii.gz "
cmd10="-t ${short_t1w}_rgd_affn_syn_on_${short_atl_t1w}_1InverseWarp.nii.gz -t [${short_t1w}_rgd_affn_syn_on_${short_atl_t1w}_0GenericAffine.mat,1]"
cmd11="-i ${atl_t2w}"

CMD3="${cmd8} ${cmd9} ${cmd10} ${cmd11}"
echo "" >> ${short_t1w}.log
echo "" >> ${short_t1w}.log
echo "Applying warp fields to images ${t1w}" >> ${short_t1w}.log
${CMD3} >> ${short_t1w}.log


