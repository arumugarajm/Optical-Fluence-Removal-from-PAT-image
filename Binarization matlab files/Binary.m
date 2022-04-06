close all;
clear;
clc

myFolder = '..\project\Dataset\1000images\0\Images'; % here give the directory of your fundus images
filePattern = fullfile(myFolder, '**/*.JPG');
theFiles = dir(filePattern);
for k = 1:length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    imageArray = imread(fullFileName);
%     ImgName=strcat('D:\project\Images\',Images(k).name);
    Img=imageArray;
    sz=256;
    I = imresize(Img, [sz,sz]);%resize the input image
    % figure, imshow(I);title('Input retina image');
    input = rgb2gray(I);%change the input image into gray image
    [rowim,colim]=size(I);

    segmented_image = segmentRetina(input);%generate the binary image 
%     S = imcomplement(segmented_image);%generate complement of binary image
% 
%     r1 = 0.01 + (0.02-0.01)*rand(rowim,rowim); %create a random matrix of size 200X200 between 0.5-0.6
%     r2 = 0.0005 + (0.0008-0.0005)*rand(rowim,rowim);%create a random matrix of size 200X200 between 0.01-0.02
% 
%     x1=segmented_image.*r1;%generate random nature for vasculature
%     x2=S.*r2;%generate random nature for background
%     Xim=x1+x2;%add both the random nature to get the final image
    save(['C:\Users\arumugaraj\Desktop\New folder\New folder\X_' num2str(k) '.mat'],'segmented_image'); % store binary images as .mat file
end
