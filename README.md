

## Image-to-Image Translation on Multi-Contrast MR Images

Generative Adversarial Neural Networks (GANs), UNets, and registration based methods for T1w-to-T2w image translation.

### [[NeuroImage paper]](https://doi.org/10.1016/j.neuroimage.2022.119091)

### Code usage  
1. Prepare your dataset under the directory 'data' in the CycleGAN folder and
set dataset name to parameter 'image_folder' in model init function.
  * Directory structure on new dataset needed for training and testing:
    * data/Dataset-name/trainA
    * data/Dataset-name/trainB
    * data/Dataset-name/testA
    * data/Dataset-name/testB  

2. Train a model by:
```
python CycleGAN/CycleGAN.py
```

3. Generate synthetic images by following specifications under:
  * CycleGAN/generate_images/ReadMe.md

### Result GIFs - 304x256 pixel images  
**Left:** Input image. **Middle:** Synthetic images generated during training. **Right:** Ground truth.  
Histograms show pixel value distributions for synthetic images (blue) compared to ground truth (brown).


#### CycleGAN - T1 to T2
![](./ReadMe/gifs/CycleGAN_T2_hist.gif?)
---


#### CycleGAN - T2 to T1
![](./ReadMe/gifs/CycleGAN_T1_hist.gif)
---


