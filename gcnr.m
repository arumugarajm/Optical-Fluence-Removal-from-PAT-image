clear all;
close all;

% Read all the results as a mat file
A = load('/media/fistlab/raj/2Dlineprofile/unet.mat');
B = load('/media/fistlab/raj/2Dlineprofile/FDUnet.mat');
C = load('/media/fistlab/raj/2Dlineprofile/Ynet.mat');
D = load('/media/fistlab/raj/2Dlineprofile/FDYnet.mat');
E = load('/media/fistlab/raj/2Dlineprofile/resnet.mat');
F = load('/media/fistlab/raj/2Dlineprofile/gan.mat');
a = B.P1;

ac = a(167:187, 55:57, :); %Selecting foreground region
figure; imshow(ac,[]);

bc = a(82:102, 205:207, :); %Selectong background region
figure; imshow(bc,[]);


%% Overlap
img_size = size(ac); %Find image size
img_size = img_size(1)*img_size(2); %To normalize the value find multiplication of rows and columns

min_img = min(min(a)); %Minimum value
max_img = max(max(a)); %maximum value
bins = linspace(min_img, max_img, 100); %create pins

figure;
temp_h1 = histogram(ac, bins); %plot histogram of foreground 

hold on
temp_h2 = histogram(bc, bins); %plot histogram of background 


h1 = temp_h1.Values;
h2 = temp_h2.Values;

h1 = h1 / img_size;
h2 = h2 / img_size;

hist_diff = zeros(1, length(h1));

for i = 1:length(h1)
   hist_diff(i) = min(h1(i), h2(i)); 
end

hist_ovl = sum(hist_diff);

gCNR = 1 - hist_ovl;
