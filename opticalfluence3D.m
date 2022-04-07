clear all;
addpath '/media/fistlab/DATA/breast dataset/iso2mesh-1.9.6-allinone/iso2mesh-1.9.6'
addpath '/media/fistlab/DATA/breast dataset/NIRFAST-9.1'
addpath '/media/fistlab/DATA/Optical-Fluence-Removal-from-PAT-image-main/k-wave-toolbox-version-1.3/k-Wave'
% load('/media/fistlab/DATA/breast dataset/715303298D_label.mat');
load('/media/fistlab/DATA/breast dataset/A.mat');
figure; imshow(label(:,:,600),[]);
% label = volimage;
% volimage1 = label;
volimage = imresize3(label,[64 64 64]);
% % 
% % % save_mesh(mesh,'/media/fistlab/raj/3D/raj');
% % 
% %  
[node,elem,face]=vol2mesh(volimage>0.005,1:size(volimage,1),1:size(volimage,2),...
                           1:size(volimage,3),0.08,0.08,1);

% % plotmesh(node,face);
% % axis equal
% % 
% % 
MyImage = label;
NewImage = zeros(size(MyImage));
[row,column,depth]=size(MyImage);

 for d=1:depth
    for r=1:row 
        for c=1:column
           if  MyImage(r,c,d) == 0
               NewImage(r,c,d)=0;
%            elseif MyImage(r,c,d) == 1
%                NewImage(r,c,d)=0.016;
           elseif MyImage(r,c,d) == 2
               NewImage(r,c,d)=0.016;
           elseif MyImage(r,c,d)==3
               NewImage(r,c,d)=0.024;
           elseif MyImage(r,c,d)==4
               NewImage(r,c,d)=0.032;
           elseif MyImage(r,c,d)==5
               NewImage(r,c,d)=0.04;
           else
               NewImage(r,c,d)=MyImage(r,c,d);
               
           end
        end 
    end
 end


 NewImage = imfill(NewImage); %To remove boundary from the dataset
 figure;imshow(NewImage(:,:,600),[]);
% % 
% % %  min1 = min(min(min(NewImage)));
% % %  max1 = max(max(max(NewImage)));
% % % volumeViewer(NewImage);
%% Edited by me
[row,col]=size(node);
Xim = imresize3(NewImage,[64 64 64]);
[ROW,COL,WID] = size(Xim);
X=node(:,1);% seperating x 
 Y=node(:,2);% seperating y
 Z=node(:,3);
 E1=ceil(X);E2=floor(X);% finding min and max values of X & Y coorinate matrices
 F1=ceil(Y);F2=floor(Y);
 G1=ceil(Z);G2=floor(Z);
 

  for i=1:row   % ceil will make index of matrix to 0 to avoid change to 1
     if E1(i)==0
         E1(i)=1;
     end
     if E2(i)==0
         E2(i)=1;
     end
     if F1(i)==0
         F1(i)=1;
     end
     if F2(i)==0
         F2(i)=1;
     end
     if G1(i)==0
         G1(i)=1;
     end
     if G2(i)==0
         G2(i)=1;
     end
  end
 
  
  for i=1:row  % floor will make index of matrix to greater than row max to avoid change to 1
     if E1(i)>ROW     %different for loop used for unequal 3D dimension
         E1(i)=ROW;
     end
     if E2(i)>ROW
         E2(i)=ROW;
     end
     if F1(i)>COL
         F1(i)=COL;
     end
     if F2(i)>COL
         F2(i)=COL;
     end
      if G1(i)>WID
         G1(i)=WID;
      end
     if G2(i)>WID
         G2(i)=WID;
     end
  end
%  
%  E1(24)=1;E1(25)=2;E1(26)=3;      E2(19)=1;E2(20)=2;E2(21)=64;
%  F1(10)=1;F1(11)=64;    F2(10)=63;F2(11)=64;
%     G2(9)=64;
%  
%   
 
 D1=zeros(row,1);% assigning matrix sizes before work with for loop
 D2=zeros(row,1);
 D3=zeros(row,1);
 D4=zeros(row,1);
 D5=zeros(row,1);
 D6=zeros(row,1);
 D7=zeros(row,1);
 D8=zeros(row,1);
 
pxmin=min(node(:,1)); % min & max values in X & Y coorinates
pymin=min(node(:,2));
pzmin=min(node(:,3));
pxmax=max(node(:,1));
pymax=max(node(:,2));
pzmax=min(node(:,3));

for i=1:row
D1(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F1(i))^2 +(Z(i)-G1(i))^2 );%formula for distance measurement
D2(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F1(i))^2+(Z(i)-G2(i))^2 );
D3(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F2(i))^2+(Z(i)-G1(i))^2 );
D4(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F2(i))^2+(Z(i)-G2(i))^2 );
D5(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F1(i))^2 +(Z(i)-G1(i))^2 );
D6(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F1(i))^2+(Z(i)-G2(i))^2 );
D7(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F2(i))^2+(Z(i)-G1(i))^2 );
D8(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F2(i))^2+(Z(i)-G2(i))^2 );
end

Ximn=zeros(row,1);%To store the pixel values as a vector
o=zeros(row,1);%To store the boundary vertex
for i=1:row  % checking which point is nearer to the original image index postion
if (D1(i)<D2(i))&&(D1(i)<D3(i))&&(D1(i)<D4(i))&&(D1(i)<D5(i))&&(D1(i)<D6(i))&&(D1(i)<D7(i))&&(D1(i)<D8(i))
    Ximn(i) = Xim(E1(i),F1(i),G1(i));
elseif(D2(i)<D1(i))&&(D2(i)<D3(i))&&(D2(i)<D4(i))&&(D2(i)<D5(i))&&(D2(i)<D6(i))&&(D2(i)<D7(i))&&(D2(i)<D8(i))
    Ximn(i) = Xim(E1(i),F1(i),G2(i));
elseif(D3(i)<D1(i))&&(D3(i)<D2(i))&&(D3(i)<D4(i))&&(D3(i)<D5(i))&&(D3(i)<D6(i))&&(D3(i)<D7(i))&&(D3(i)<D8(i))
    Ximn(i) = Xim(E1(i),F2(i),G1(i));
elseif(D4(i)<D1(i))&&(D4(i)<D2(i))&&(D4(i)<D3(i))&&(D4(i)<D5(i))&&(D4(i)<D6(i))&&(D4(i)<D7(i))&&(D4(i)<D8(i))
    Ximn(i) = Xim(E1(i),F2(i),G2(i));
elseif(D5(i)<D1(i))&&(D5(i)<D2(i))&&(D5(i)<D3(i))&&(D5(i)<D4(i))&&(D5(i)<D6(i))&&(D5(i)<D7(i))&&(D5(i)<D8(i))
    Ximn(i) = Xim(E2(i),F1(i),G1(i));
elseif(D6(i)<D1(i))&&(D6(i)<D2(i))&&(D6(i)<D3(i))&&(D6(i)<D5(i))&&(D6(i)<D4(i))&&(D6(i)<D7(i))&&(D6(i)<D8(i))
    Ximn(i) = Xim(E2(i),F1(i),G2(i));
elseif(D7(i)<D1(i))&&(D7(i)<D2(i))&&(D7(i)<D3(i))&&(D7(i)<D5(i))&&(D7(i)<D6(i))&&(D7(i)<D4(i))&&(D7(i)<D8(i))
    Ximn(i) = Xim(E2(i),F2(i),G1(i));
elseif(D8(i)<D1(i))&&(D8(i)<D2(i))&&(D8(i)<D3(i))&&(D8(i)<D5(i))&&(D8(i)<D6(i))&&(D8(i)<D7(i))&&(D8(i)<D4(i))
    Ximn(i) = Xim(E2(i),F2(i),G2(i));
 else
     Ximn(i) =  Xim(E1(i),F2(i),G1(i));
%      Ximn(i) =  0.02;
end

if (pxmin==node(i,1)) % finding boundary vertex. just finding extreme points in the rectangle
    o(i)=1;
elseif(pymin==node(i,2))
    o(i)=1;
elseif(pzmin==node(i,3))
    o(i)=1;
elseif (pxmax==node(i,1))
    o(i)=1;
elseif (pymax==node(i,2))
    o(i)=1;
elseif(pzmax==node(i,3))
    o(i)=1;
end
end
% Ximn(row+1:end,:)=[];
% u=zeros(row,1);


% clear all;
% load('/home/fistlab/Desktop/3D/m.mat');
vertn=0.1*[X Y Z];% generating node points according to NIRFAST
% vertn=[X Y Z];


ksi=0.2129*ones(row,1);
c=2.255*10^11*ones(row,1);
kappa=0.33*ones(row,1);
ri=1.33*ones(row,1);
mus=ones(row,1);
num=[1;2;3;4;5;6;7;8;9;10];%10 sources
% num=[1;2;3;4;5;6;7;8;9;10;11;12];%12 sources


coord = 0.1*[60.2362 30.2695 34.3414; 56.4456 19.387 34.9734; 52.4662 16.3406 41.9984; 46.7856 17.2916 49.5006; 40.6312 11.5002 48.4067; 34.2673 9.53353 48.5871; 28.151 6.50021 47.4434; 22.7605 7.49991 49.2403; 17.9932 5.99991 47.7709; 12.9238 6.49993 49.1318]; %A
% coord = 0.1*[60.2502 30.233 37.7255; 57.753 28.5209 46.9161; 53.1039 28.0716 52.5386; 46.4664 29.0118 56.395; 39.8266 24.609 57.7508; 34.9362 29.0966 59.7501; 25.8753 25.6024 60.8344; 19.7172 25.7598 61.5006; 12.3572 25.6485 62.5005; 8.66866 27.8311 62.9997]; %B
% coord = 0.1*[58.1252 27.0824 40.0106; 55.6442 27.7318 49.0221; 50.4857 26.8831 53.1426; 44.6876 27.0376 54.6137; 36.8105 23.0113 55.5926; 31.4093 20.2714 58.2511; 26.6511 20.5537 59.6254; 21.7216 19.7153 60.2505; 14.3531 15.3454 58.4895; 11.5113 26.7128 62.6124]; %C
% coord = 0.1*[58.2506 29.9428 38.8656; 56.751 32.347 44.5591; 52.501 25.082 45.7649; 44.7111 17.0937 46.2371; 42.5575 25.8201 52.8662; 36.414 26.6518 56.8466; 31.0475 26.6889 58.7101; 22.0116 19.6862 57.1718; 19.1026 26.8515 60.7514; 14.7382 25.2195 59.6904];%D

% coord = 0.1*[6.22528 40.7932 30.7266; 9.14643 40.3288 38.5745; 15.0271 39.4798 46.17; 21.4156 33.302 54.0228; 28.2838 39.4015 56.8346; 34.4998 39.3472 60.214; 41.6927 39.0483 62.0001; 48.49991 42.6632 62.3111; 54.1221 44.9336 62.5001; 60.6116 50.7802 61.5007]; %07
% coord = 0.1*[5.99987 30.0224 38.1414; 11.4953 33.7381 43.7746; 18.0234...
% 36.0105 47.0929; 22.9827 38.4343 49.4995; 30.7575 39.7689 52.0008;...
% 39.9337 37.659 56.0002; 43.8368 42.7543 55.7468; 47.9865 42.3508 57.5004;...
% 54.0607 37.1889 60.4999; 59.4997 43.2802 60.3118];%35
% coord = 0.1*[4.25006 33.723 26.4916; 8.52471 32.2638 34.7688; 17.807 36.7012 42.3997; 27.8135 39.1871 47.8413; 33.5295 39.4136 50.9995; 39.9358 36.3486 55.0002; 42.2903 37.0902 56.8026; 51.2567 37.2192 58.2258; 56.6384 38.9409 59.5005; 61.7352 40.7645 60.5005]; %47
fwhm=zeros(10,1);%for 10 sources
% fwhm=ones(10,1);%for 10 sources


v=[1;1;1;1;1;1;1;1;1;...
    2;2;2;2;2;2;2;2;2;...
    3;3;3;3;3;3;3;3;3;...
    4;4;4;4;4;4;4;4;4;...
    5;5;5;5;5;5;5;5;5;...
    6;6;6;6;6;6;6;6;6;...
    7;7;7;7;7;7;7;7;7;...
    8;8;8;8;8;8;8;8;8;...
    9;9;9;9;9;9;9;9;9;...
    10;10;10;10;10;10;10;10;10];
w=[2;3;4;5;6;7;8;9;10;...
    3;4;5;6;7;8;9;10;1;...
    4;5;6;7;8;9;10;1;2;...
    5;6;7;8;9;10;1;2;3;...
    6;7;8;9;10;1;2;3;4;...
    7;8;9;10;1;2;3;4;5;...
    8;9;10;1;2;3;4;5;6;...
    9;10;1;2;3;4;5;6;7;...
    10;1;2;3;4;5;6;7;8;...
    1;2;3;4;5;6;7;8;9];

% x=ones(100,1);
x=ones(90,1);
% x=ones(144,1);

link=[v w x];% link for source and detector
% o = ones(row,1);

%generating mesh, source, meas structures
source.distributed=0;
source.fixed=1;
source.num=num;
source.coord=coord;
source.fwhm=fwhm;

mesh.name='cylinder';
mesh.nodes=vertn(:,1:3);
mesh.bndvtx=o;
mesh.type='stnd';
mesh.mua=Ximn;
% disp(mesh.mua);
% mesh.muan=Ximn;
mesh.kappa=kappa;
mesh.ri=ri;
mesh.mus=mus;
mesh.elements=elem(:,1:4);
% mesh.elements=face(:,1:4);
mesh.dimension=3;
mesh.source=source;
mesh.link=link;
mesh.c=c;
mesh.ksi=ksi;

meas.fixed=1;
meas.num=[1;2;3;4;5;6;7;8;9;10];
% meas.num=[1;2;3;4;5;6;7;8;9;10;11;12];


meas.coord = 0.1*[60.2362 30.2695 34.3414; 56.4456 19.387 34.9734; 52.4662 16.3406 41.9984; 46.7856 17.2916 49.5006; 40.6312 11.5002 48.4067; 34.2673 9.53353 48.5871; 28.151 6.50021 47.4434; 22.7605 7.49991 49.2403; 17.9932 5.99991 47.7709; 12.9238 6.49993 49.1318]; %A
% meas.coord = 0.1*[60.2502 30.233 37.7255; 57.753 28.5209 46.9161; 53.1039 28.0716 52.5386; 46.4664 29.0118 56.395; 39.8266 24.609 57.7508; 34.9362 29.0966 59.7501; 25.8753 25.6024 60.8344; 19.7172 25.7598 61.5006; 12.3572 25.6485 62.5005; 8.66866 27.8311 62.9997]; %B
% meas.coord = 0.1*[58.1252 27.0824 40.0106; 55.6442 27.7318 49.0221; 50.4857 26.8831 53.1426; 44.6876 27.0376 54.6137; 36.8105 23.0113 55.5926; 31.4093 20.2714 58.2511; 26.6511 20.5537 59.6254; 21.7216 19.7153 60.2505; 14.3531 15.3454 58.4895; 11.5113 26.7128 62.6124]; %C
% meas.coord = 0.1*[58.2506 29.9428 38.8656; 56.751 32.347 44.5591; 52.501 25.082 45.7649; 44.7111 17.0937 46.2371; 42.5575 25.8201 52.8662; 36.414 26.6518 56.8466; 31.0475 26.6889 58.7101; 22.0116 19.6862 57.1718; 19.1026 26.8515 60.7514; 14.7382 25.2195 59.6904];%D


% meas.coord = 0.1*[6.22528 40.7932 30.7266; 9.14643 40.3288 38.5745; 15.0271 39.4798 46.17; 21.4156 33.302 54.0228; 28.2838 39.4015 56.8346; 34.4998 39.3472 60.214; 41.6927 39.0483 62.0001; 48.49991 42.6632 62.3111; 54.1221 44.9336 62.5001; 60.6116 50.7802 61.5007]; %07
% meas.coord = 0.1*[5.99987 30.0224 38.1414; 11.4953 33.7381 43.7746; 18.0234...
% 36.0105 47.0929; 22.9827 38.4343 49.4995; 30.7575 39.7689 52.0008;...
% 39.9337 37.659 56.0002; 43.8368 42.7543 55.7468; 47.9865 42.3508 57.5004;...
% 54.0607 37.1889 60.4999; 59.4997 43.2802 60.3118];%35
% meas.coord = 0.1*[4.25006 33.723 26.4916; 8.52471 32.2638 34.7688; 17.807 36.7012 42.3997; 27.8135 39.1871 47.8413; 33.5295 39.4136 50.9995; 39.9358 36.3486 55.0002; 42.2903 37.0902 56.8026; 51.2567 37.2192 58.2258; 56.6384 38.9409 59.5005; 61.7352 40.7645 60.5005]; %47

[ind, int_func] = mytsearchn(mesh,mesh.source.coord);
mesh.meas=meas;

r=[ind int_func];
mesh.meas.int_func=r;

% save(['/home/fistlab/Desktop/3D/m.mat'],'X','Y','Z','elem', 'Ximn');


data = femdata(mesh,0);


S = sum(data.phi,2);% add all the fluence generated by the source.  It add all the columns
format long
S;
% min(S);
% S1=zeros(row,1);
S1=full(S);% checking sum and full produce the same result or not


% plotimage(mesh,S1);%plotimage function will plot image for a single variable
new=mesh.mua.*S1;%mua and fluence multiplication
% new=mesh.mua;
% figure; imshow(new,[]);


% plotmesh(mesh,new);%plot image for new mua value

mesh.mua=new;%changing mua value to mua*phi
% plotmesh(mesh);%plotting mesh with optical fluence

% k = boundary(Ximn);


 D1n=zeros(row,1);% assigning matrix sizes before work with for loop
 D2n=zeros(row,1);
 D3n=zeros(row,1);
 D4n=zeros(row,1);
 D5n=zeros(row,1);
 D6n=zeros(row,1);
 D7n=zeros(row,1);
 D8n=zeros(row,1);
 
 
for i=1:row
D1n(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F1(i))^2 +(Z(i)-G1(i))^2 );%formula for distance measurement
D2n(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F1(i))^2+(Z(i)-G2(i))^2 );
D3n(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F2(i))^2+(Z(i)-G1(i))^2 );
D4n(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F2(i))^2+(Z(i)-G2(i))^2 );
D5n(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F1(i))^2 +(Z(i)-G1(i))^2 );
D6n(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F1(i))^2+(Z(i)-G2(i))^2 );
D7n(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F2(i))^2+(Z(i)-G1(i))^2 );
D8n(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F2(i))^2+(Z(i)-G2(i))^2 );
end

Ximg=zeros(ROW,COL,WID);

for i=1:row  % checking which point is nearer to the original image index postion
if (D1n(i)<D2n(i))&&(D1n(i)<D3n(i))&&(D1n(i)<D4n(i))&&(D1n(i)<D5n(i))&&(D1n(i)<D6n(i))&&(D1n(i)<D7n(i))&&(D1n(i)<D8n(i))
     Ximg(E1(i),F1(i),G1(i))=mesh.mua(i);
elseif(D2n(i)<D1n(i))&&(D2n(i)<D3n(i))&&(D2n(i)<D4n(i))&&(D2n(i)<D5n(i))&&(D2n(i)<D6n(i))&&(D2n(i)<D7n(i))&&(D2n(i)<D8n(i))
    Ximg(E1(i),F1(i),G2(i))=mesh.mua(i);
elseif(D3n(i)<D1n(i))&&(D3n(i)<D2n(i))&&(D3n(i)<D4n(i))&&(D3n(i)<D5n(i))&&(D3n(i)<D6n(i))&&(D3n(i)<D7n(i))&&(D3n(i)<D8n(i))
    Ximg(E1(i),F2(i),G1(i))=mesh.mua(i);
elseif(D4n(i)<D1n(i))&&(D4n(i)<D2n(i))&&(D4n(i)<D3n(i))&&(D4n(i)<D5n(i))&&(D4n(i)<D6n(i))&&(D4n(i)<D7n(i))&&(D4n(i)<D8n(i))
    Ximg(E1(i),F2(i),G2(i))=mesh.mua(i);
elseif(D5n(i)<D1n(i))&&(D5n(i)<D2n(i))&&(D5n(i)<D3n(i))&&(D5n(i)<D4n(i))&&(D5n(i)<D6n(i))&&(D5n(i)<D7n(i))&&(D5n(i)<D8n(i))
    Ximg(E2(i),F1(i),G1(i))=mesh.mua(i);
elseif(D6n(i)<D1n(i))&&(D6n(i)<D2n(i))&&(D6n(i)<D3n(i))&&(D6n(i)<D5n(i))&&(D6n(i)<D4n(i))&&(D6n(i)<D7n(i))&&(D6n(i)<D8n(i))
    Ximg(E2(i),F1(i),G2(i))=mesh.mua(i);
elseif(D7n(i)<D1n(i))&&(D7n(i)<D2n(i))&&(D7n(i)<D3n(i))&&(D7n(i)<D5n(i))&&(D7n(i)<D6n(i))&&(D7n(i)<D4n(i))&&(D7n(i)<D8n(i))
    Ximg(E2(i),F2(i),G1(i))=mesh.mua(i);
elseif(D8n(i)<D1n(i))&&(D8n(i)<D2n(i))&&(D8n(i)<D3n(i))&&(D8n(i)<D5n(i))&&(D8n(i)<D6n(i))&&(D8n(i)<D7n(i))&&(D8n(i)<D4n(i))
    Ximg(E2(i),F2(i),G2(i))=mesh.mua(i);
else
     Ximg(E2(i),F1(i),G2(i))=mesh.mua(i);
%      Ximg(E2(i),F1(i),G2(i))=mesh.mua(i);

end
end
volumeViewer(Xim); %volumeviewer app
% % % volumeViewer(Ximg); %volumeviewer app
% % 
% Ximgn = imfill(Ximg);
Ximgf = imresize3(Ximg,[128 128 128]);  
imshow(Ximgf(:,:,50),[]);
% 
clear source;

% create the computational grid
PML_size = 10;              % size of the PML in grid points
Nx = 150 - 2 * PML_size;     % number of grid points in the x direction
Ny = Nx;                    % number of grid points in the y direction
Nz = Nx;                    % number of grid points in the z direction
x = 25.1e-3;
y = x;
z = x;
dx = x/(Nx-1); %0.2e-3;                % grid point spacing in the x direction [m]
dy = dx;                    % grid point spacing in the y direction [m]
dz = dx;                    % grid point spacing in the z direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

% define the properties of the propagation medium
medium.sound_speed = 1540;	% [m/s]


M = 64;
N = 64;
P = 64;
indxi = ceil(Nx/2) - M:ceil(Nx/2) + M -1;
indyi = ceil(Ny/2) - N:ceil(Ny/2) + N -1;
indzi = ceil(Nz/2) - P:ceil(Nz/2) + P -1;

Nxi = length(indxi);
Nyi = length(indyi);
Nzi = length(indzi);
a = Ximgf;

% 
% figure;imshow(a(:,:,40),[]);
% 
p0_binary = zeros(Nx, Ny, Nz);
p0_binary(indxi,indyi,indzi) = a(:,:,:);


% 
% smooth the initial pressure distribution and restore the magnitude
p0 = smooth(p0_binary, true);
% 
% 
% assign to the source structure
source.p0 = p0;
% 

% 
center_freq = 5e6;      % [Hz]
bandwidth = 90;        % [%]
sensor.frequency_response = [center_freq, bandwidth];
% % 
% % define a binary planar sensor
% A = (linspace(1,128,58));
% A = round(linspace(1,128,64));
A = linspace(2,128,64);
s1 = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
s1(A, A, 1) = 1;

s4 = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
s4(A, 128, A) = 1;


s2 = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
s2(A, 1, A) = 1;

s3 = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
s3(1, A, A) = 1;


sensor.mask =  s1+s2+s3+s4;

% create the time array
[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed);
kgrid.t_array = 0:1e-8:2048*0.3*dt;

% kgrid.makeTime(medium.sound_speed);

% set the input arguements
input_args = {'PMLSize', PML_size, 'PMLInside', false, ...
    'PlotPML', false, 'Smooth', false, 'DataCast', 'gpuArray-double'};

% run the simulation
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});

signal_to_noise_ratio = 40;	% [dB]

sdn2 = addNoise(sensor_data, signal_to_noise_ratio, 'peak');
% 
% 
% reset the initial pressure
source.p0 = 0;

% assign the time reversal data
sensor.time_reversal_boundary_data = sdn2;%sensor_data;

% run the time-reversal reconstruction
p0_recon = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
p0_recon = gather(p0_recon);
sdn2 = gather(sdn2);
sdn = sdn2;
sensor_data = gather(sensor_data);
% % 
% % % % add first order compensation for only recording over a half plane
Ximg1 = 2 * p0_recon;
% 
% % apply a positivity condition
Ximg1(Ximg1 < 0) = 0;
% % imshow(sensor_data);
% % %
Ximg1 = imresize3(Ximg1,[128 128 128]);
slice1 = 90;
figure;imshow(p0_binary(:,:,slice1),[]);
figure;imshow(Ximg1(:,:,slice1),[]);
Xgt = imresize3(NewImage,[128 128 128]);
% 
save('/media/fistlab/DATA/breast dataset/kwavenew/47.mat','p0_recon','sdn','Ximg','Ximgf','Ximg1','NewImage');
% % 
% % % % 
% % create the computational grid
% PML_size = 10;              % size of the PML in grid points
% Nx = 148+2 - 2 * PML_size;     % number of grid points in the x direction
% Ny = Nx;                    % number of grid points in the y direction
% Nz = Nx;                    % number of grid points in the z direction
% x = 25.1e-3;
% y = x;
% z = x;
% dx = x/(Nx-1); %0.2e-3;                % grid point spacing in the x direction [m]
% dy = dx;                    % grid point spacing in the y direction [m]
% dz = dx;                    % grid point spacing in the z direction [m]
% kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
% 
% % define the properties of the propagation medium
% medium.sound_speed = 1500;	% [m/s]
% 
% 
% M = 64;
% N = 64;
% P = 64;
% indxi = ceil(Nx/2) - M:ceil(Nx/2) + M -1;
% indyi = ceil(Ny/2) - N:ceil(Ny/2) + N -1;
% indzi = ceil(Nz/2) - P:ceil(Nz/2) + P -1;
% 
% Nxi = length(indxi);
% Nyi = length(indyi);
% Nzi = length(indzi);
% a = Ximgf;
% 
% % 
% % figure;imshow(a(:,:,40),[]);
% % 
% p0_binary = zeros(Nx, Ny, Nz);
% p0_binary(indxi,indyi,indzi) = a(:,:,:);
% 
% 
% % 
% % smooth the initial pressure distribution and restore the magnitude
% p0 = smooth(p0_binary, true);
% % 
% % 
% % assign to the source structure
% source.p0 = p0;
% % 
% 
% % 
% center_freq = 5e6;      % [Hz]
% bandwidth = 90;        % [%]
% sensor.frequency_response = [center_freq, bandwidth];
% % 
% % % define a binary planar sensor
% % A = (linspace(1,128,58));
% % A = 1:2:128;
% s1 = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
% s1(1:2:128, 1:2:128, 1) = 1;
% 
% s4 = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
% s4(A, 128, A) = 1;
% % imshow(s4(:,:,2));
% 
% s2 = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
% s2(3:2:126, 1, 3:2:126) = 1;
% 
% s3 = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
% s3(1, A, A) = 1;
% 
% 
% sensor.mask =  s1+s2;
% 
% % create the time array
% [kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed);
% kgrid.t_array = 0:1e-8:2048*0.3*dt;
% 
% % kgrid.makeTime(medium.sound_speed);
% 
% % set the input arguements
% input_args = {'PMLSize', PML_size, 'PMLInside', false, ...
%     'PlotPML', false, 'Smooth', false, 'DataCast', 'gpuArray-single'};
% 
% % run the simulation
% sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
% 
% signal_to_noise_ratio = 40;	% [dB]
% 
% sdn2 = addNoise(sensor_data, signal_to_noise_ratio, 'peak');
% % 
% % 
% % reset the initial pressure
% source.p0 = 0;
% 
% % assign the time reversal data
% sensor.time_reversal_boundary_data = sdn2;%sensor_data;
% 
% % run the time-reversal reconstruction
% p0_recon = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
% p0_recon = gather(p0_recon);
% sdn2 = gather(sdn2);
% sensor_data = gather(sensor_data);
% % % 
% % % % % add first order compensation for only recording over a half plane
% Ximg1 = 2 * p0_recon;
% % 
% % % apply a positivity condition
% Ximg1(Ximg1 < 0) = 0;
% % % imshow(sensor_data);
% % % %
% slice1 = 80;
% figure;imshow(p0_binary(:,:,slice1),[]);
% figure;imshow(Ximg1(:,:,slice1),[]);
% % % voxelPlot(double(sensor.mask));
