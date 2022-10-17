# Optical-Fluence-Removal-from-PAT-images
## Introduction

## 2D model:
For 2D models, use the 2D directory and run the Optical fulence.m file, it requires im2mesh function, NIRFAST, and K-Wave toolbox. This file has optical forward problem, acoustic forward and acoustic inverse problem solutions in 2D.
For the 2D model, we used fundus image dataset to generate the vasculature nature for PAT images.
Fundus image datasets are available in online [Dataset 1](https://www.kaggle.com/linchundan/fundusimage1000).
[Datset 2](https://www5.cs.fau.de/research/data/fundus-images/). 
im2mesh function link is here [im2mesh](https://in.mathworks.com/matlabcentral/fileexchange/71772-im2mesh-2d-image-to-triangular-meshes), which is used to generate 2D mesh & also addpath for [mesh2d-master](https://in.mathworks.com/matlabcentral/fileexchange/25555-mesh2d-delaunay-based-unstructured-mesh-generation). The NIRFAST toolbox link is here: [NIRAST](https://milab.host.dartmouth.edu/nirfast/).
The K-Wave toolbox link is here: [K-Wave](http://www.k-wave.org/).


## 3D model:
Use the 3D directory and run the opticalfluence3D.m file to solve the optical forward problem, acoustic forward problem, and inverse problem in 3D. For the 3D model, we collected online datasets from the following links [Dataset 1](https://anastasio.bioengineering.illinois.edu/downloadable-content/oa-breast-database/) and [Dataset 2](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KBYQQ7).
The iso2mesh function can be download from this link [iso2mesh](http://iso2mesh.sourceforge.net/cgi-bin/index.cgi), which is used to generate 3D mesh and also we need open-source K-Wave and NIRFAST toolboxes, and the link is given above. 


## DL model:
The DL models directory has all the Deep Learning models(U-Net, FD U-Net, Y-Net, FD Y-Net, Deep ResU-Net, and GAN). If you have any questions, please contact me at: arumugaraj.m@gmail.com


## Dataset preparation
To generate a dataset, the following operations were performed on fundus images. The resized, binarized, and optical absorption coefficient of tissue value were assigned.


## Optical forward problem
This process is done with the help of the NIRFAST toolbox. We already knew the mua value. Now we have to give the remaining properties to the NIRFAST toolbox and generate the optical energy distribution.


## Acoustic forward and inverse problem
We know the initial pressure distribution. Now we assign all the acoustic properties in the K-Wave toolbox and get the measured time series data. 
With the inversion of the data, we can get the K-Wave reconstructed image. This will be the input to the deep learning model.


## Deep learning model
To remove optical fluence, we used different DL models. The parameters and hyperparameters are: Adam optimizer, loss function = average normalized mean squared error, batch size = 6

## How to use codes
### Step 1: Generation of binary phantom
For 2D, first generate binary images from Fundus images by using Binarization Matlab files. This file has 4 matlab files. Here binary.m contains code for binarazation and the remaining files support this main program. Note: this step can be skipped if we directly follow the next step.
### Step 2: Generation of realistic phantom and forward optical, forward and inverse acoustic models or fluence corrupted image generation
For 2D, fluence affected images can be generated simply by running the Optical fluence.m file. To run this code we need some supporting codes from Binarization matlab files (imoverlay.m,isodat.m,segmentRetina.m),forward.m, inverse.m, and NIRFAST toolbox. 
For 3D, fluence affected images can be generated simply by running the opticalfluence3D.m file and removing unnecessary tissues from the original phantom. This can be done by using a simple for loop and changing all tissue values. 
Store all the images (2D and 3D) as 2D mat files to use for DL models. To avoid a limited number of datasets, a 3D volume image must be converted to a 2D slice. 
### Step 3: Train, Validate, and Test DL models or removal of fluence effect using DL models:
The DL models directory has all the DL models. Split the dataset into three parts, 80% used for training, 10% used for validation, and 10% used for testing the DL models. Run train. py to train and store the best weight of the model and load these weights to test the DL model's performance.



