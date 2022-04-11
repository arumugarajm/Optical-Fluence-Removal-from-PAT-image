# Optical-Fluence-Removal-from-PAT-images
## Introduction
## 2D model:
For 2D model, use 2D directory and run the Optical fulence.m file, it required im2mesh function, NIRFAST, and K-Wave toolbox & this file has optical forward problem, acoustic forward and inverse problem solution in 2D.
For 2D model we used fundus image dataset to generate the vasculature nature for PAT images.
Fundus image datasets are available in online [Dataset 1](https://www.kaggle.com/linchundan/fundusimage1000).
[Datset 2](https://www5.cs.fau.de/research/data/fundus-images/). 
im2mesh function link is here [im2mesh](https://in.mathworks.com/matlabcentral/fileexchange/71772-im2mesh-2d-image-to-triangular-meshes), which is used to generate 2D mesh & also addpath for [mesh2d-master](https://in.mathworks.com/matlabcentral/fileexchange/25555-mesh2d-delaunay-based-unstructured-mesh-generation).NIRFAST toolbox link is here [NIRAST](https://milab.host.dartmouth.edu/nirfast/).
K-Wave toolbox link is here [K-Wave](http://www.k-wave.org/).
## 3D model:
For 3D model, use 3D directory and run opticalfluence3D.m file & this file has optical forward problem, acoustic forward and inverse problem solution for 3D.For 3D model we collected online datasets from the following links [Dataset 1](https://anastasio.bioengineering.illinois.edu/downloadable-content/oa-breast-database/) and [Dataset 2](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KBYQQ7).
 iso2mesh function can download from this link [iso2mesh](http://iso2mesh.sourceforge.net/cgi-bin/index.cgi), which is used to generate 3D mesh and also we need open-source K-Wave and NIRFAST toolboxes and the link is given above. 
## DL model:
DL models directory has all the Deep learning models(U-Net, FD U-Net, Y-Net, FD Y-Net, Deep ResU-Net, and GAN).
For more details refer our presentation [link for ppt](https://docs.google.com/presentation/d/11RDKK24GpVCmXNCmAEXCF1me9gWeFw93/edit#slide=id.p1). For any questions mail me: arumugaraj.m@gmail.com


## Dataset preparation
To generate dataset, the following operations were performed on fundus images. Resized,Binarized and Optical absorption coefficient of tissue value was assigned.


## Optical forward problem
This process done with the help of NIRFAST toolbox. Already we knew mua value, now we have to give remaining properties to NIRFAST toolbox and the generate the Optical energy distribution.


## Acoustic forward and inverse problem
We knew initial pressure distribution, now assign all the acoustic properties in K-Wave toolbox and get the measured time series data.
Take the inversion of the data we can get the K-Wave reconstructed image. This will be the input to the Deep learnig model.


## Deep learning model
To remove optical fluence, we used different DL models. The parameters and hyperparameters are: Adam optimizer, loss function = average normalized mean squared error, batch size = 6

## How to use codes
### Step 1: Generation of binary phantom
For 2D, first generate binary images from Fundus images by using Binarization matlab files. This flie has 4 matlab files, here binary.m conatin code for binarazation and the remaining files are supporting to this main program. Note: this step can skip if we directly use next step.
### Step 2: Generation of realistic phantom and forward optical, forward and inverse acoustic models or fluence corrupted image generation
For 2D, fluence affected images can be generated simply by running Optical fluence.m file, to run this code we need some supporting codes from Binarization matlab files (imoverlay.m,isodat.m,segmentRetina.m),forward.m,inverse.m,NIRFAST toolbox.
For 3D, fluence affected images can be generated simply by running opticalfluence3D.m file and remove unnecessary tissues from the original phantom, this can be done by using simple for loop and change all tissue values.
Store all the images (2D and 3D) as 2D mat files to use for DL models. Note, 3D volume image must convert to 2D slice to avoid limited number of dataset.
### Step 3: Train, Validate, and Test DL models or removal of fluence effect using DL models:
DL models directory has all the DL models. Split the dataset into two folders, one should have 90% of total data and other have 10% of total data, do data-augmentaion. Run train.py to train and store best weight of the model and load these weights to test the DL model performance.



