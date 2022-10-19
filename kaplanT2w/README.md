# kaplanT2w

This software uses ANTs and FSL to generate a pseudo-T2w image from a T1w image in neonatal MRI.

The currrent paper is at https://doi.org/10.1016/j.neuroimage.2022.119091

# Software Requirements
* ANTs
  * Install from: http://stnava.github.io/ANTs/
* FSL
  * Install from: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

# Set up
All image files are expected to be in nifti format.

## Configuration File ##
Sets paths to atlases, scripts, and toolkits. Additional parameters related to the atlases are included. A template configuration file is included.

`script_dir` This is the path to these downloaded scripts

`ANTSPATH` This is the path to where the ANTs toolkit binaries are installed

`atl_dir` This is the path to where the atlases are located

`age_bins` These are the age bins of the atlases. See "Atlases" below for more information.

`atl_ids` These are the participant ids of the atlases. See "Atlases" below for more information.

`bin_idx` These indicate the bin that a participant belongs to based on the corresponding index in `age_bins`.

## Atlases ##
High-quality T1w and T2w images that have been co-registered within an individual atlas participant, as well as the skull-stripped versions of these images. Recommended at least 10 individual atlas participants per age bin. Recommended atlas images have been N4 corrected. Age bins are organized by gestational age in weeks. The number of bins as well as ages contained in a bin are optional. Naming conventions for files and age bins must be organized in the following structure:

<!-- AUTO-GENERATED-CONTENT:START (DIRTREE:dir=./&depth=1) -->
```
atl_dir/
├── minage1_maxage1/
    ├── PATIENT01_T1w.nii.gz
    ├── PATIENT01_T1w_brain.nii.gz
    ├── PATIENT01_T2w.nii.gz
    ├── PATIENT01_T2w_brain.nii.gz
    ├── PATIENT02_T1w.nii.gz
    ├── PATIENT02_T1w_brain.nii.gz
    ├── PATIENT02_T2w.nii.gz
    └── PATIENT02_T2w_brain.nii.gz
├── minage2_maxage2/
    ├── PATIENT03_T1w.nii.gz
    ├── PATIENT03_T1w_brain.nii.gz
    ├── PATIENT03_T2w.nii.gz
    ├── PATIENT03_T2w_brain.nii.gz
    ├── PATIENT04_T1w.nii.gz
    ├── PATIENT04_T1w_brain.nii.gz
    ├── PATIENT04_T2w.nii.gz
    └── PATIENT04_T2w_brain.nii.gz
└── minage3_maxage3/
    ├── PATIENT05_T1w.nii.gz
    ├── PATIENT05_T1w_brain.nii.gz
    ├── PATIENT05_T2w.nii.gz
    ├── PATIENT05_T2w_brain.nii.gz
    ├── PATIENT06_T1w.nii.gz
    ├── PATIENT06_T1w_brain.nii.gz
    ├── PATIENT06_T2w.nii.gz
    └── PATIENT06_T2w_brain.nii.gz
```
<!-- AUTO-GENERATED-CONTENT:END -->

***NOTE:*** Expected directory name structure for atlases is age ranges in weeks of the bins separated by an underscore.
Ex: /atl_dir/38_40 --> this bin is for ages 38-40 weeks

## Running ##
The main processing script is called `generate_kaplanT2w.csh` and can be run as:

`<path-to-scripts>/generate_kaplanT2w.csh -i <t1w nifti> -c <config file> [options]`

### Processing directives and options ###
* REQUIRED:
  * `-i <input image>`: Full path to your T1w nifti image
  * `-c <config file>`: Full path to your configuration file
* OPTIONAL:
  * `-a <age>`: Gestational age in weeks (default: 40)
  * `-o <output image>`: Output kaplan-T2w nifti image (default: input_kaplanT2w.nii.gz)
  * `-m <brain mask>`: Full path to brain mask nifti image
  * `-d <output dir>`: Full path to output directory
  * `-n <1/0>`: apply ANTs N4 bias field correction (default: 1)
  * `-s`: save intermediary files (off by default)
