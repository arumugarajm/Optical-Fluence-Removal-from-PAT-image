myFolder = '../Dataset/Binary_img';
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
    load(fullFileName);
    Y = imcomplement(X);
    r = 0.01*randn(256,256);
    r = imgaussfilt(r, 0.7);
    Y1 = Y.*r;

    ind = find(X(:)==1);

    a = max(max(r));
    a = 0.3*a;
    r(ind) = a;

    X2 = (X+1).*r;
    
    X1 = X2*10+0.45;
    GT = X1+Y1;
%     save(['../Dataset/Train_data_MSOT/Ground_truth/GT_' num2str(k) '.mat'],'GT');
    
%     RT1 = Recon1_tikhonov(:,:,15);
%     RT2 = Recon1_tikhonov1(:,:,15);
    
    figure
%     subplot(1,2,1)
    histogram(GT);
% %     subplot(1,3,2)
% %     histogram(RT1);
%     subplot(1,2,2)
%     histogram(RT2);
    
    figure
%     subplot(1,2,1)
    imshow(GT, [])
% %     subplot(1,3,2)
% %     imshow(RT1, [])
%     subplot(1,2,2)
%     imshow(RT2, [])
end