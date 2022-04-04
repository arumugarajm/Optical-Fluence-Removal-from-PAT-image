# Optical-Fluence-Removal-from-PAT-images
## Introduction
For this project we used fundus image dataset to generate the vasculature nature for PAT images.
Fundus image datasets are available here [Dataset 1](https://www.kaggle.com/linchundan/fundusimage1000).
[Datset 2](https://www5.cs.fau.de/research/data/fundus-images/). For 3D model we collected online datasets [Dataset 1](https://anastasio.bioengineering.illinois.edu/downloadable-content/oa-breast-database/) and [Dataset 2](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KBYQQ7).
For 2D model run the Optical fulence.m file, it required im2mesh function, NIRFAST, and K-Wave toolbox this file has optical forward problem, acoustic forward and inverse problem solution in 2D.
im2mesh function link is here [im2mesh](https://in.mathworks.com/matlabcentral/fileexchange/71772-im2mesh-2d-image-to-triangular-meshes). For 3D model run 3D for acoustic phantom file this file has optical forward problem, acoustic forward and inverse problem solution for 3D.
NIRFAST toolbox link is here [NIRAST](https://milab.host.dartmouth.edu/nirfast/).
K-Wave toolbox link is here [K-Wave](http://www.k-wave.org/).
For binarization operation we need binarization matlab files.
For more details refer our presentation [link for ppt](https://docs.google.com/presentation/d/11RDKK24GpVCmXNCmAEXCF1me9gWeFw93/edit#slide=id.p1). For any questions mail me: arumugaraj.m@gmail.com


## Dataset preparation
To generate dataset, the following operations were performed on fundus images. Resized,Binarized and Optical absorption coefficient of tissue value was assigned


## Optical forward problem
This process done with the help of NIRFAST toolbox. Already we knew mua value, now we have to give remaining properties to NIRFAST toolbox and the generate the Optical energy distribution


## Acoustic forward and inverse problem
We knew initial pressure distribution, now assign all the acoustic properties in K-Wave toolbox and get the measured time series data.
Take the inversion of the data we can get the K-Wave reconstructed image. This will be the input to the Deep learnig model.


## Deep learning model
To remove optical fluence, we used U-Net model. The parameters are: Adam optimizer, loss function = average normalized mean squared error, batch size = 30
