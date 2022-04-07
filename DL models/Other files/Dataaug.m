%% this program conatain the flipping operation
clc;
clear all;
myFolder = '/home/fistlab/Desktop/3D/Presentation/2D/47';
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
% filePattern = fullfile(myFolder, '**\*.JPG');
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
for k = 1:length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
%     fprintf(1, 'Now reading %s\n', fullFileName);
   a =  load(fullFileName);
% a = load('/media/fistlab/raj/combined dataset/X2_1.mat');
% b = a.Xi;
b = a.Xi;
c = a.Xgt;
% d = a.sdn;
e = a.sdn;
% figure;imshow(Xi,[]);
% figure;imshow(Xim,[]);
%%Data augmenation
% Xi = flip(b);
% Xgt = flip(c);
% % sdn = flip(d);
% sdn = flip(e);


Xi = flip(b,2);
Xgt = flip(c,2);
% sdn = flip(d);
sdn = flip(e,2);

save(['/home/fistlab/Desktop/3D/Presentation/2D/47/X47F2_' num2str(k) '.mat'],'Xgt','Xi','sdn');
end








% %% this program conatain the flipping operation and remove complete zero
% intensity images
% clc;
% clear all;
% myFolder = '/media/fistlab/DATA/breast dataset/kwave result/2D';
% if ~isdir(myFolder)
%   errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
%   uiwait(warndlg(errorMessage));
%   return;
% end
% % filePattern = fullfile(myFolder, '**\*.JPG');
% filePattern = fullfile(myFolder, '*.mat');
% theFiles = dir(filePattern);
% for k = 1:length(theFiles)
%     baseFileName = theFiles(k).name;
%     fullFileName = fullfile(theFiles(k).folder, baseFileName);
% %     fprintf(1, 'Now reading %s\n', fullFileName);
%    a =  load(fullFileName);
% % a = load('/media/fistlab/raj/combined dataset/X2_1.mat');
% % b = a.Xi;
% b = a.Xi;
% % c = a.Xf;
% % d = a.sdn;
% e = a.Xgt;
% minv = min(min(e));
% maxv = max(max(e));
% if maxv-minv ~=0
%     Xi = b;
% %     Xf = c;
% %     sdn = e;
%     Xgt = e;
%     save(['/media/fistlab/DATA/breast dataset/kwave result/2D/a/F2_' num2str(k) '.mat'],'Xi','Xgt');
% end
%     
% % figure;imshow(Xi,[]);
% % figure;imshow(Xim,[]);
% 
% end

% clear all;
% % convert 3D data into 2D slices
% L = load('/home/fistlab/Desktop/3D/Presentation/471.mat');
% % L = load('D:\3D\rat data\X.mat');
% O1 = L.Xin;
% O2 = L.Xgt;
% O3 = L.sdn;
% [R,C,W] = size(O2);
% 
% for i=1:128
% %     minv = min(min(O2));
% %     maxv = max(max(O2));
% %     if maxv-minv ~=0
%         Xi = O1(:,:,i);
%         Xgt = O2(:,:,i);
%         sdn = O3;
%     
%         save(['/home/fistlab/Desktop/3D/Presentation/2D/47/X47_' num2str(i) '.mat'],'Xgt','Xi', 'sdn');
%     end
% % end

