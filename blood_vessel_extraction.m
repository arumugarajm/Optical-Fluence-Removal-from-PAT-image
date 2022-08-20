clear; 
Images = dir('/media/fistlab/DATA/Raj/Dataset/0/*.JPG'); % read all fundus image dataset from the directory % Find all jpg files in current directory.

for k=1:length(Images)
    baseFileName = Images(k).name;
    fullFileName = fullfile(Images(k).folder, baseFileName);
    Img = imread(fullFileName);
    sz=256;
    I = imresize(Img, [sz,sz]);%resize the input image
    input = rgb2gray(I);%change the input image into gray image
    [rowim,colim]=size(input);

    segmented_image = segmentRetina(input);%generate the binary image 
%     figure, imshow(segmented_image);
    S = imcomplement(segmented_image);%generate complement of binary image

%     r1 = 0.03 + (0.0438-0.03)*rand(rowim,rowim); %create a random matrix of size 200X200 between 0.5-0.6
%     r2 = 0.0005 + (0.0008-0.0005)*rand(rowim,rowim);%create a random matrix of size 200X200 between 0.01-0.02
    r1 = 0.03 + (0.04-0.03)*rand(rowim,rowim); %create a random matrix of size 200X200 between 0.5-0.6
    r2 = 0 + (0.001-0)*rand(rowim,rowim);%create a random matrix of size 200X200 between 0.01-0.02
    

    x1=segmented_image.*r1;%generate random nature for vasculature
    x2=S.*r2;%generate random nature for background
    GT=x1+x2;%add both the random nature to get the final image
%     figure;imshow(Xim,[]);
%     save(['C:\Users\arumugaraj\Desktop\New folder (5)\X\X_' num2str(k) '.mat'],'Xim');
%     imwrite(Img,['D:\project\Dataset\All image\X_' num2str(i) '.jpg']);
