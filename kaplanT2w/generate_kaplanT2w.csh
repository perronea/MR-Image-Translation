#!/bin/csh

@ numIn = $#argv # number of input arguments

set program = $0; set program = $program:t

#############
### usage ###
#############
if ($numIn < 4) then
    echo "Usage:    " $program "-i <t1w nifti> -c <config file> [options]"
    echo ""
    echo "REQUIRED:"
    echo "-i <t1w nifti>    input t1w nifti image"
    echo "-c <config file>  configuration file"
    echo ""
    echo "OPTIONS:"
    echo "-a <age>          gestational age in weeks (default: 40)"
    echo "-o <out nifti>    kaplanT2w nifti image (default: input_kaplanT2w.nii.gz)"
    echo "-m <mask>         brain mask nifti image"
    echo "-d <output dir>   desired output directory"
    echo "-n <1/0>          apply N4 correction (default: 1)"
    echo "-s                save intermediary files (off by default)"
    exit
endif


##################
### get inputs ###
##################
@ i = 1
@ j = 2
set rois = ""
while ($i <= $numIn)
    switch ($argv[$i])
        case -i:
            set t1wfullpath = $argv[$j]
            breaksw;
        case -c:
            set config = $argv[$j]
            source $config
            breaksw;
        case -a:
            set age = $argv[$j]
            breaksw;
        case -o:
            set kapt2wfullpath = $argv[$j]
            breaksw;
        case -m:
            set brainmask = $argv[$j]
            @ needmask = 0
            breaksw;
        case -d:
            set outdir = $argv[$j]
            set workdir = ${outdir}/kaplanT2w
            breaksw;
        case -n:
            @ n4 = $argv[$j]
        case -s:
            set del_intermed = 0;
            breaksw;
        default:
            echo "ERROR: Unrecognized option: "$argv[$i]
            exit
            breaksw;
    endsw
    @ i = $i + 2
    @ j = $j + 2
end

#################################
### check for required fields ###
#################################
if (! $?t1wfullpath) then
    echo "ERROR: -i input image required"
    exit
endif

if (! $?config) then
    echo "ERROR: -c configuration file required"
    exit
endif

####################
### set defaults ###
####################
set curdir = `pwd`
set t1w = `echo $t1wfullpath | rev | cut -d "/" -f1 | rev | cut -d "." -f1`

if (! $?outdir) then
    set outdir = ${curdir}
    set workdir = ${curdir}/kaplanT2w
endif

if ($t1wfullpath == `basename $t1wfullpath`) then
    set t1wfullpath = ${curdir}/$t1wfullpath
endif


if (! $?age) then
    set age = 40
endif

if (! $?kapt2wfullpath) then
    set kapt2wfullpath = ${outdir}/${t1w}_kaplanT2w.nii.gz
else if ($kapt2wfullpath == `basename $kapt2wfullpath`) then
    set kapt2wfullpath = ${outdir}/$kapt2wfullpath
endif

if (! $?del_intermed) then
    set del_intermed = 1;
endif

if (! $?n4) then
    @ n4 = 1
endif

if (! $?brainmask) then
    set brainmask = ${workdir}/${t1w}_brain_mask.nii.gz
    @ needmask = 1
else
    if ($brainmask == `basename $brainmask`) then
        set brainmask = ${curdir}/$brainmask
    endif
endif

##########################
### set up directories ###
##########################
if (! -d $outdir) mkdir $outdir
if (! -d $workdir) mkdir $workdir

###################
### get age bin ###
###################
set age_min = ()
set age_max = ()
foreach weekNum ( $age_bins )
    set age_min = ($age_min `echo $weekNum | cut -d "_" -f1`)
    set age_max = ($age_max `echo $weekNum | cut -d "_" -f2`)
end

@ i = 1
@ older = 1
while ( $older && $i < = $#age_max )
    if ($age <= $age_max[$i] ) @ older = 0
    @ i++
end
@ binNum = $i - 1
set weekNum = $age_bins[$binNum]

echo ""
echo "AGE BIN IS "$weekNum
echo ""

pushd $workdir
#################################
### get atlas ids for age bin ###
#################################
if (-e atl_ids.txt) then
    rm atl_ids.txt
    touch atl_ids.txt
endif
@ i = 1
foreach sub ( $atl_ids )
    printf "${sub}\t${bin_idx[$i]}\n" >> atl_ids.txt
    @ i++
end
set atl_idxs = `awk '{print $2}' atl_ids.txt | grep $binNum -n | awk -F":" '{print $1}'`
set atl_ids_bin = ()
foreach idx ($atl_idxs)
    set atl_ids_bin = ($atl_ids_bin `head -$idx atl_ids.txt | tail -1 | awk '{print $1}'`)
end
rm atl_ids.txt



###########################
### apply n4 correction ###
###########################
if ($n4) then
    echo ""
    echo "RUNNING N4 CORRECTION..."
    echo ""
    ${ANTSPATH}/N4BiasFieldCorrection -d 3 --input-image $t1wfullpath --shrink-factor 2 --output ${t1w}_N4.nii.gz
    set t1wfullpath = $workdir/${t1w}_N4.nii.gz
    set t1w = ${t1w}_N4
    if ($needmask) set brainmask = ${workdir}/${t1w}_brain_mask.nii.gz
endif

if (! -d body_reg) mkdir body_reg
pushd body_reg
###########################################
### register banks subjects to get body ###
###########################################
echo ""
echo "BODY REGISTRATION..."
foreach sub ($atl_ids_bin)
    echo "REGISTERING SUBJECT: "$sub
    echo ""
    # perform ANTs registration
    ${script_dir}/ANTS_reg_init.sh $t1wfullpath $atl_dir/$weekNum/${sub}_T1w.nii.gz $atl_dir/$weekNum/${sub}_T2w.nii.gz $ANTSPATH
end

#################################
### fuse body images together ###
#################################
set imgs = ""
foreach sub ($atl_ids_bin)
    # normalize T2s from 0-3000
    set minval = `fslstats ${sub}_T2w_on_${t1w}_init.nii.gz -R | awk '{print $1}'`
    set maxval = `fslstats ${sub}_T2w_on_${t1w}_init.nii.gz -R | awk '{print $2}'`
    fslmaths ${sub}_T2w_on_${t1w}_init.nii.gz -sub $minval -div `echo $maxval $minval | awk '{print $1-$2}'` -mul 3000 ${sub}_T2w_on_${t1w}_init_norm
	    
    # normalize T1s from 0-1400
    set minval = `fslstats ${sub}_T1w_on_${t1w}_init.nii.gz -R | awk '{print $1}'`
    set maxval = `fslstats ${sub}_T1w_on_${t1w}_init.nii.gz -R | awk '{print $2}'`
    fslmaths ${sub}_T1w_on_${t1w}_init.nii.gz -sub $minval -div `echo $maxval $minval | awk '{print $1-$2}'` -mul 1400 ${sub}_T1w_on_${t1w}_init_norm
        
    # set up output string
    set imgs = `echo $imgs" -g ["$sub"_T1w_on_"$t1w"_init_norm.nii.gz, "$sub"_T2w_on_"$t1w"_init_norm.nii.gz]"`
end

# apply joint fusion to t2 images
echo ""
echo "FUSING BODY IMAGES..."
echo ""

$ANTSPATH/antsJointFusion -t $t1wfullpath $imgs -o ${t1w}_T2w_JF_body_init.nii.gz
mv ${t1w}_T2w_JF_body_init.nii.gz ../
popd # out of body_reg

if ($needmask) then
    # get brain mask
    echo ""
    echo "GETTING BRAIN MASK..."
    echo ""
    bet2 ${t1w}_T2w_JF_body_init.nii.gz ${t1w}_brain -g -0.15 -r 45 -m
endif

echo ""
echo "APPLYING BRAIN MASK TO T1..."
echo ""
# apply brain mask to t1
fslmaths $t1wfullpath -mas $brainmask ${t1w}_brain

if (! -d brain_reg) mkdir brain_reg
pushd brain_reg
echo ""
echo "BRAIN REGISTRATION..."
foreach sub ($atl_ids_bin)
    echo "REGISTERING SUBJECT: "$sub
    echo "  INTITIAL ANTS REGISTRATION..."
    echo ""
    # perform ANTs registration
    ${script_dir}/ANTS_reg_init.sh $workdir/${t1w}_brain.nii.gz $atl_dir/$weekNum/${sub}_T1w_brain.nii.gz $atl_dir/$weekNum/${sub}_T2w_brain.nii.gz $ANTSPATH

    echo ""
    echo "  FINAL ANTS REGISTRATION..."
    echo ""
    # perform final ANTs registration
    ${script_dir}/ANTS_reg_final.sh $workdir/${t1w}_brain.nii.gz ${sub}_T1w_brain_on_${t1w}_brain_init.nii.gz ${sub}_T2w_brain_on_${t1w}_brain_init.nii.gz $ANTSPATH
end

##################################
### fuse brain images together ###
##################################
set imgs = ""
foreach sub ($atl_ids_bin)
    # normalize T2s from 0-3000
    set minval = `fslstats ${sub}_T2w_brain_on_${t1w}_brain_final.nii.gz -R | awk '{print $1}'`
    set maxval = `fslstats ${sub}_T2w_brain_on_${t1w}_brain_final.nii.gz -R | awk '{print $2}'`
    fslmaths ${sub}_T2w_brain_on_${t1w}_brain_final.nii.gz -sub $minval -div `echo $maxval $minval | awk '{print $1-$2}'` -mul 3000 ${sub}_T2w_brain_on_${t1w}_brain_final_norm
	    
    # normalize T1s from 0-1400
    set minval = `fslstats ${sub}_T1w_brain_on_${t1w}_brain_final.nii.gz -R | awk '{print $1}'`
    set maxval = `fslstats ${sub}_T1w_brain_on_${t1w}_brain_final.nii.gz -R | awk '{print $2}'`
    fslmaths ${sub}_T1w_brain_on_${t1w}_brain_final.nii.gz -sub $minval -div `echo $maxval $minval | awk '{print $1-$2}'` -mul 1400 ${sub}_T1w_brain_on_${t1w}_brain_final_norm
        
    # set up output string
    set imgs = `echo $imgs" -g ["$sub"_T1w_brain_on_"$t1w"_brain_final_norm.nii.gz, "$sub"_T2w_brain_on_"$t1w"_brain_final_norm.nii.gz]"`
end

# apply joint fusion to t2 images
echo ""
echo "FUSING BRAIN IMAGES..."
echo ""

$ANTSPATH/antsJointFusion -t $workdir/${t1w}_brain.nii.gz $imgs -o ${t1w}_T2w_JF_brain.nii.gz
mv ${t1w}_T2w_JF_brain.nii.gz ../
popd # out of brain_reg

###########################
### denoise fused brain ###
###########################
echo ""
echo "DENOISING IMAGE..."
echo ""
$ANTSPATH/DenoiseImage -d 3 -n Rician --input-image ${t1w}_T2w_JF_brain.nii.gz --output ${t1w}_T2w_JF_brain_dn.nii.gz

############################
# perform histogram matching
############################
# mask image
fslmaths ${t1w}_T2w_JF_brain_dn.nii.gz -mas $brainmask ${t1w}_T2w_JF_brain_dn_brain.nii.gz
set kapT2 = $workdir/${t1w}_T2w_JF_brain_dn_brain.nii.gz

if (! -d hist_match) mkdir hist_match
pushd hist_match

# histogram match each subject
echo ""
echo "PERFORMING HISTOGRAM MATCHING..."
@ i = 1
foreach sub ($atl_ids_bin)
    echo "      Matching "$sub"..."
    # perform histogram matching
    ${ANTSPATH}/ImageMath 3 ${t1w}_brain_histmatch_${sub}.nii.gz HistogramMatch $kapT2 $atl_dir/$weekNum/${sub}_T2w_brain.nii.gz 1024 7 1
    
    # create sum image
    if ($i == 1) then
        cp ${t1w}_brain_histmatch_${sub}.nii.gz ${t1w}_brain_histmatch.nii.gz
    else
        fslmaths ${t1w}_brain_histmatch.nii.gz -add ${t1w}_brain_histmatch_${sub}.nii.gz ${t1w}_brain_histmatch
    endif
    @ i++
end

# create average
fslmaths ${t1w}_brain_histmatch.nii.gz -div $#atl_ids_bin ${t1w}_brain_histmatch

# create normalized image 0-2500
echo ""
echo ""
echo "PERFORMING NORMALIZATION..."
echo ""
set minval = `fslstats ${t1w}_brain_histmatch.nii.gz -R | awk '{print $1}'`
set maxval = `fslstats ${t1w}_brain_histmatch.nii.gz -R | awk '{print $2}'`
set range = `echo $minval $maxval | awk '{print $2 - $1}'`
fslmaths ${t1w}_brain_histmatch.nii.gz -sub $minval -div $range -mul 2500 ${t1w}_brain_histmatch_norm
mv ${t1w}_brain_histmatch_norm.nii.gz ../
popd # out of hist_match

##############################
### combine body and brain ###
##############################
echo ""
echo "EXTRACTING BODY ONLY..."
echo ""
set maskname = `echo $brainmask | rev | cut -d "/" -f1 | rev | cut -d "." -f1`
fslmaths $brainmask -binv $maskname"_inv"
fslmaths ${t1w}_T2w_JF_body_init.nii.gz -mas ${maskname}_inv ${t1w}_body_only.nii.gz

echo ""
echo "COMBINING BRAIN AND BODY..."
echo ""
fslmaths ${t1w}_brain_histmatch_norm.nii.gz -add ${t1w}_body_only.nii.gz $kapt2wfullpath

echo ""
echo "KAPLAN-T2W COMPLETE AT: "$kapt2wfullpath
echo ""

popd # out of workdir

if ($del_intermed) then
    mv $workdir/${t1w}_brain_mask.nii.gz $outdir
    rm -r $workdir
endif


