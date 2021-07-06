# Optical-Fluence-Removal-from-PAT-image
## Introduction
For this project we used fundus image dataset to generate the vasculature nature for PAT images.
Fundus image dataset is available here [Dataset](https://www.kaggle.com/linchundan/fundusimage1000).
[Datset](https://www5.cs.fau.de/research/data/fundus-images/).
To run the Optical fulence.m file, it required im2mesh function, NIRFAST, and K-Wave toolbox.
im2mesh function link is here [im2mesh](https://in.mathworks.com/matlabcentral/fileexchange/71772-im2mesh-2d-image-to-triangular-meshes).
NIRFAST toolbox link is here [NIRAST](https://milab.host.dartmouth.edu/nirfast/).
K-Wave toolbox link is here [K-Wave](http://www.k-wave.org/).
### Dataset preparation
To generate dataset, the following operations performed on fundus images. Resized,Binarized and Optical absorption coefficient value is assigned
#### Optical forward problem
This process done with the help of NIRFAST toolbox. Already we knew mua value, now we have to give remaining properties to NIRFAST toolbox and the generate the Optical energy distribution
##### Acoustic forward problem
We knew initial pressure distribution, now assign all the acoustic properties in K-Wave toolbox and get the measured time series data.
Take the inversion of the data we can get the K-Wave reconstructed image. This will be the input to the Deep learnig model.
###### Deep learning model
To remove optical fluence, we use U-Net model. The parameters are: Adam optimizer, loss function = average normalized ean squared error, batch size = 30
