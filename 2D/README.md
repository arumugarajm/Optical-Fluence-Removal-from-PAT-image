This directory has 3 files.
Optical fluence.m is a main file to generate the 2D fluence corrupted image. This file need sub-program files from Binarization matlab files and forward, inverse matlab files, and NIRFAST toolbox.
forward.m is used to run K-Wave forward model.
inverse.m is used to run K-Wave inverse model.
Note: We can use inbuilt function files from K-Wave toolbox. If you have GPU access then change input arguments in forward and inverse matlab files to gpu-Array it will accelerate the computation speed.
