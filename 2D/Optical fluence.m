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
    addpath('mesh2d-master')    % for poly2mesh
    
    % ---------------------------------------------------------------------
    % Example 2
    % Demonstrate function im2mesh, and export mesh as inp file (Abaqus), 
    % bdf file (Nastran bulk data), and .node/.ele file.
    % ---------------------------------------------------------------------
    % part 1: obtain mesh
    
    %import grayscale segmented image
    im=segmented_image;%It take only binary image as the input

    % parameters
    tf_avoid_sharp_corner = true;  % Whether to avoid sharp corner
    tolerance =1.3;                 % Tolerance for polygon simplification
    hmax = 1.1;                     % Maximum mesh-size
    mesh_kind = 'delfront';         % Meshing algorithm
    grad_limit = 0.3;             % Scalar gradient-limit for mesh
    
    select_phase = [];  % Parameter type: vector
                        % If 'select_phase' is [], all the phases will be
                        % chosen.
                        % 'select_phase' is an index vector for sorted 
                        % grayscales (ascending order) in an image.
                        % For example, an image with grayscales of 40, 90,
                        % 200, 240, 255. If u're interested in 40, 200, and
                        % 240, then set 'select_phase' as [1 3 4]. Those 
                        % phases corresponding to grayscales of 40, 200, 
                        % and 240 will be chosen to perform meshing.
                        
    
    % function im2mesh includes the operations shown in demo1.m
    [ vert,tria,tnum ] = im2mesh( im, select_phase, tf_avoid_sharp_corner, tolerance, hmax, mesh_kind, grad_limit );
    
    
    
    % ---------------------------------------------------------------------
    % part 2: write inp, bdf, and .node/.ele file 
    
    % parameters for mesh export
    dx = 1; dy = 1;     % scale of your imgage
                        % dx - column direction, dy - row direction
                        % e.g. scale of your imgage is 0.11 mm/pixel, try
                        %      dx = 0.11; and dy = 0.11;
                        
    ele_order = 1;      % for getNodeEle, this variable is either 1 or 2
                        % 1 - linear / first-order element
                        % 2 - quadratic / second-order element
                        % note: printBdf only support linear element
                        
    ele_type = 'CPS3';  % element type, for printInp
    
    precision_nodecoor = 8; % precision of node coordinates, for printInp
                            % e.g. precision_nodecoor=4, dx=1 -> 0.5000 
                            %      precision_nodecoor=3, dx=0.111, -> 0.055
    
    % scale node coordinates
    vert( :, 1 ) = vert( :, 1 ) * dx;
    vert( :, 2 ) = vert( :, 2 ) * dy;
    
    % get node coordinares and elements from mesh
    [ nodecoor_list, nodecoor_cell, ele_cell ] = getNodeEle( vert, tria, ...
                                                        tnum, ele_order );
    
    % write file
    % inp file (Abaqus)
    % print as multiple parts
    printInp_multiPart( nodecoor_cell, ele_cell, ele_type, precision_nodecoor );
    % print as multiple sections
    printInp_multiSect( nodecoor_list, ele_cell, ele_type, precision_nodecoor );
    
    % bdf file (Nastran bulk data)
    printBdf( nodecoor_list, ele_cell, precision_nodecoor );
    
    % .node/.ele file 
    % haven't been tested
    printTria( vert, tria, tnum, precision_nodecoor )
    
%     plotMeshes( vert, tria, tnum );
    
     
 Z=vert;
 [row,col]=size(Z);%getting size of the vertex matrix
%  Znew=reshape(Z,79998,1);
%  X=Znew((1:39999),:);
%  Y=Znew((40000:79998),:);
 Znew=reshape(Z,row*2,1);% reshaping the vertex matrix
 X=Znew((1:row),:);% seperating x coordinates from Z
 Y=Znew((row+1:end),:);% seperating y coordinates from Z
 E1=ceil(X);E2=floor(X);% finding min and max values of X & Y coorinate matrices
 F1=ceil(Y);F2=floor(Y);
 
 for i=1:row   % ceil will make index of matrix to 0 to avoid change to 1
     if E1(i)==0
         E1(i)=1;
     elseif E2(i)==0
         E2(i)=1;
     end
     if F1(i)==0
         F1(i)=1;
     elseif F2(i)==0
         F2(i)=1;
     end
 end
 
  for i=1:row   % floor will make index of matrix to greater than row max to avoid change to 1
     if E1(i)>rowim
         E1(i)=rowim;
     elseif E2(i)>rowim
         E2(i)=rowim;
     end
     if F1(i)>rowim
         F1(i)=rowim;
     elseif F2(i)>rowim
         F2(i)=rowim;
     end
 end
 D1=zeros(row,1);% assigning matrix sizes before work with for loop
 D2=zeros(row,1);
 D3=zeros(row,1);
 D4=zeros(row,1);
 
pxmin=min(vert(:,1)); % min & max values in X & Y coorinates
pymin=min(vert(:,2));
pxmax=max(vert(:,1));
pymax=max(vert(:,2));
for i=1:row
D1(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F1(i))^2);%formula for distance measurement
D2(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F2(i))^2);
D3(i)=sqrt((X(i)-E1(i))^2+(Y(i)-F2(i))^2);
D4(i)=sqrt((X(i)-E2(i))^2+(Y(i)-F1(i))^2);
end
Ximn=zeros(row,1);%To store the pixel values as a vector
o=zeros(row,1);%To store the boundary vertex
for i=1:row  % checking which point is nearer to the original image index postion
if (D1(i)<D2(i))&&(D1(i)<D3(i))&&(D1(i)<D4(i))
    %Xn(i)=E1(i);Yn(i)=F1(i);
   Ximn(i) = GT(E1(i),F1(i));
elseif(D2(i)<D1(i))&&(D2(i)<D3(i))&&(D2(i)<D4(i))
   %Xn(i)=E2(i);Yn(i)=F2(i);
   Ximn(i) = GT(E2(i),F2(i));
elseif(D3(i)<D1(i))&&(D3(i)<D2(i))&&(D3(i)<D4(i))
   %Xn(i)=E1(i);Yn(i)=F2(i);
   Ximn(i) = GT(E1(i),F2(i));
elseif(D4(i)<D1(i))&&(D4(i)<D2(i))&&(D4(i)<D3(i))
    %Xn(i)=E2(i);Yn(i)=F1(i);
    Ximn(i) = GT(E2(i),F1(i));
% elseif(D1(i)==D2(i))&&(D2(i)==D3(i))&&(D3(i)==D4(i))
%     Ximn(i) = Xim(E1(i),F2(i));
 else
     Ximn(i) =  GT(E2(i),F1(i));%Xim(E1(i),F2(i));
end
if (pxmin==vert(i,1)) % finding boundary vertex. just finding extreme points in the rectangle
    o(i)=1;
elseif(pymin==vert(i,2))
    o(i)=1;
elseif (pxmax==vert(i,1))
    o(i)=1;
elseif (pymax==vert(i,2))
    o(i)=1;
end
end

Ximn(row+1:end,:)=[];% removing unnecessary mua from the original image
o(row+1:end,:)=[];% removing unnecessary extreme points from the boundary vertex matrix
u=zeros(row,1);% in NIRFAST third coordinate values set to zero for 2D
%vertn=[vert u];
%vertn=[Y -X u];
vertn=0.5*0.5*0.125*[X Y u];% generating node points according to NIRFAST


ksi=0.2129*ones(row,1);
c=2.255*10^11*ones(row,1);
kappa=0.33*ones(row,1);
ri=1.33*ones(row,1);
mus=ones(row,1);
num=[1;2;3;4;5;6;7;8;9;10];%10 sources
% num=[1;2;3;4;5;6;7;8;9;10;11;12];%12 sources

% coord=[0.5 0.5;0.5 10.4;0.5 21.4;0.5 31.3;0.5 39;0.5 51;0.5 60.9;0.5 70.8;0.5 80.7;0.5 90.6];%100x100
%coord=[0.5 0.5;0.5 10.4;0.5 21.4;0.5 31.3;0.5 39;0.5 48.6;0.5 56.3;0.5 65.1;0.5 75;0.5 79.4];%80x80
coord=0.5*0.5*0.5*[0.125 0.4;0.125 5.075;0.125 10.3; 0.125 14.975; 0.125 20.2;0.125 25.125; 0.125 30.05;0.125 35;0.125 39.95;0.125 45.175];%50x50
fwhm=zeros(10,1);%for 10 sources
% fwhm=zeros(12,1);%for 12 sources


v=[1;1;1;1;1;1;1;1;1;1;...
    2;2;2;2;2;2;2;2;2;2;...
    3;3;3;3;3;3;3;3;3;3;...
    4;4;4;4;4;4;4;4;4;4;...
    5;5;5;5;5;5;5;5;5;5;...
    6;6;6;6;6;6;6;6;6;6;...
    7;7;7;7;7;7;7;7;7;7;...
    8;8;8;8;8;8;8;8;8;8;...
    9;9;9;9;9;9;9;9;9;9;...
    10;10;10;10;10;10;10;10;10;10];
w=[1;2;3;4;5;6;7;8;9;10;...
    1;2;3;4;5;6;7;8;9;10;...
    1;2;3;4;5;6;7;8;9;10;...
    1;2;3;4;5;6;7;8;9;10;...
    1;2;3;4;5;6;7;8;9;10;...
    1;2;3;4;5;6;7;8;9;10;...
    1;2;3;4;5;6;7;8;9;10;...
    1;2;3;4;5;6;7;8;9;10;...
    1;2;3;4;5;6;7;8;9;10;...
    1;2;3;4;5;6;7;8;9;10];

x=ones(100,1);
% x=ones(144,1);

link=[v w x];% link for source and detector


%generating mesh, source, meas structures
source.distributed=0;
source.fixed=1;
source.num=num;
source.coord=coord;
source.fwhm=fwhm;

mesh.name='rectangle';
mesh.nodes=vertn;
mesh.bndvtx=o;
mesh.type='stnd';
mesh.mua=Ximn;
% disp(mesh.mua);
mesh.muan=Ximn;
mesh.kappa=kappa;
mesh.ri=ri;
mesh.mus=mus;
mesh.elements=tria;
mesh.dimension=2;
mesh.source=source;
mesh.link=link;
mesh.c=c;
mesh.ksi=ksi;

meas.fixed=1;
meas.num=[1;2;3;4;5;6;7;8;9;10];
% meas.num=[1;2;3;4;5;6;7;8;9;10;11;12];

%meas.coord=0.5*0.5*0.5*[0.125 0.4;0.125 5.075;0.125 10.3; 0.125 14.975; 0.125 20.2;0.125 25.125; 0.125 30.05;0.125 35;0.125 39.95;0.125 45.175];meas.coord=0.5*0.5*0.5*[0.125 0.4;0.125 5.075;0.125 10.3; 0.125 14.975; 0.125 20.2;0.125 25.125; 0.125 30.05;0.125 35;0.125 39.95;0.125 45.175];
%meas.coord=0.5*0.5*0.5*[0.125 0.4;0.125 5.075;0.125 10.3; 0.125 14.975; 0.125 20.2;0.125 25.125; 0.125 30.05;0.125 35;0.125 39.95;0.125 45.175];
meas.coord=[6.51563 0.015625;6.51563 0.6;6.51563 1.21875;6.51563 1.80313;6.51563 2.45625;6.51563 3.24687;6.51563 4.075;6.51563 4.79688;6.51563 5.27813;6.51563 5.79375];
[ind, int_func] = mytsearchn(mesh,mesh.source.coord);
%meas.int_func=[ind int_func];
mesh.meas=meas;
r=[ind int_func];
mesh.meas.int_func=r;

% plot mesh and generate the data for the corresponding mesh
% plotmesh(mesh,1);% inbuilt function used to plot mesh
%plotmesh1(mesh,0); %modified function to plot mesh
%saveas(gcf,'Barchart.png')
% plot(mesh.nodes(1:end,1),mesh.nodes(1:end,2),'c.');
% axis equal;
% save_mesh(mesh,meshb);
data = femdata(mesh,0);
% plotmesh1(mesh);

S = sum(data.phi,2);% add all the fluence generated by the source.  It add all the columns
format long
S;
min(S);
% S1=zeros(row,1);
S1=full(S);% checking sum and full produce the same result or not


% plotimage(mesh,S1);%plotimage function will plot image for a single variable
new=mesh.mua.*S1;%mua and fluence multiplication
% plotmesh1(mesh,new);%plot image for new mua value
mesh.mua=new;%changing mua value to mua*phi
% plotmesh1(mesh);%plotting mesh with optical fluence
% h=figure(2);
% imwrite(h,'myGray.png');
% saveas(h,'m.jpg');
% fileName = sprintf('%d.jpg'); % Create filename.
% saveas(h, fileName);

% end
% p0=loadImage('m.png');
% p1=p0(50:584,186:721);
% red=m(:,:,1); green=m(:,:,2);blue=m(:,:,3);
 D1n=zeros(row,1);% assigning matrix sizes before work with for loop
 D2n=zeros(row,1);
 D3n=zeros(row,1);
 D4n=zeros(row,1);
 
 for i=1:row
D1n(i)=sqrt((E1(i)-X(i))^2+(F1(i)-Y(i))^2);%formula for distance measurement
D2n(i)=sqrt((E2(i)-X(i))^2+(F2(i)-Y(i))^2);
D3n(i)=sqrt(E1(i)-X(i))^2+(F2(i)-(Y(i))^2);
D4n(i)=sqrt(E2(i)-(X(i))^2+(F1(i)-Y(i))^2);
end


Ximg=zeros(rowim,colim);
for i=1:row  % checking which point is nearer to the original image index postion
   if (D1n(i)<D2n(i))&&(D1n(i)<D3n(i))&&(D1n(i)<D4n(i))
    %Xn(i)=E1(i);Yn(i)=F1(i);
   Ximg(E1(i),F1(i))=mesh.mua(i);
elseif(D2n(i)<D1n(i))&&(D2n(i)<D3n(i))&&(D2n(i)<D4n(i))
   %Xn(i)=E2(i);Yn(i)=F2(i);
   Ximg(E2(i),F2(i))=mesh.mua(i);
elseif(D3n(i)<D1n(i))&&(D3n(i)<D2n(i))&&(D3n(i)<D4n(i))
   %Xn(i)=E1(i);Yn(i)=F2(i);
   Ximg(E1(i),F2(i))=mesh.mua(i);
elseif(D4n(i)<D1n(i))&&(D4n(i)<D2n(i))&&(D4n(i)<D3n(i))
    %Xn(i)=E2(i);Yn(i)=F1(i);
    Ximg(E2(i),F1(i))=mesh.mua(i);
% elseif(D1(i)==D2(i))&&(D2(i)==D3(i))&&(D3(i)==D4(i))
%     Ximn(i) = Xim(E1(i),F2(i));
 else
     Ximg(E2(i),F1(i))=mesh.mua(i);%Xim(E1(i),F2(i));
    end
end

addpath('../k_wave_PAT_CODE/k-wave-toolbox-version-1.3/k-Wave');
medium.sound_speed = 1540;  % [m/s]
% create the time array
time.dt = 5e-8;         % sampling time in sec 5e-8
time.length = 500;      % number of points in time

object_sim.Nx = 801;  % number of grid points in the x (row) direction
object_sim.Ny = 801;  % number of grid points in the y (column) direction
object_sim.x = 80.1e-3;              % total grid size [m]
object_sim.y = 80.1e-3;              % total grid size [m]

dx = object_sim.x/object_sim.Nx;              % grid point spacing in the x direction [m]
dy = object_sim.y/object_sim.Ny;              % grid point spacing in the y direction [m]
kgrid = makeGrid(object_sim.Nx, dx, object_sim.Ny, dy);

% create a second computation grid for the reconstruction to avoid the
% inverse crime
object_rec.Nx = (1/2)*(object_sim.Nx-1)+1;    % number of grid points in the x (row) direction
object_rec.Ny = (1/2)*(object_sim.Ny-1)+1;    % number of grid points in the y (column) direction
object_rec.x = (object_sim.x);              % total grid size [m]
object_rec.y = (object_sim.y);              % total grid size [m]

dx_rec = (object_rec.x/(object_rec.Nx-1));               % grid point spacing in the x direction [m]
dy_rec = (object_rec.y/(object_rec.Ny-1));              % grid point spacing in the y direction [m]
recon_grid = makeGrid(object_rec.Nx, dx_rec, object_rec.Ny, dy_rec);


% define a centered Cartesian circular sensor
clear sensor;
sensor_radius = 19e-3;     % [m]
sensor_angle = 2*pi;      % [rad]
sensor_pos = [0, 0];        % [m]
num_sensor_points = 256;
cart_sensor_mask = makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle);
sensor.mask = cart_sensor_mask;
center_freq = 5e6;      % [Hz]
bandwidth = 90;        % [%]
sensor.frequency_response = [center_freq, bandwidth];
M = 128;
N = 128;
indxi = ceil(object_sim.Nx/2) - M:ceil(object_sim.Nx/2) + M-1;
indyi = ceil(object_sim.Ny/2) - N:ceil(object_sim.Ny/2) + N-1;

Nxi = length(indxi);
Nyi = length(indyi);
tl = time.length;
sml = length(sensor.mask);
ANx = tl*sml;
ANy = Nxi*Nyi;



signal_to_noise_ratio = 40;	% [dB]
scaling_factor = 0.15;
IM_init_v = zeros((2*M+1)*(2*N+1),1);
p0_magnitude = 1;
% bv=load('D:\project\New folder\acode_fileExchange_941\new codes\reitna-segmentation-master\Ximg.mat','Ximg');
a=Ximg;
% figure;imshow(a,[]);

object_sim.p0 = zeros(object_sim.Nx, object_sim.Ny);
object_sim.p0(indxi,indyi) = a(:,:);
% figure;imshow(object_sim.p0,[]);


sd21 = forward(object_sim, time, medium, sensor);
% filter the sensor data using a Gaussian filter
sd2 = gaussianFilter(sd21, t2, center_freq, bandwidth);
% add noise to the recorded sensor data
sdn2 = addNoise(sd2, signal_to_noise_ratio, 'peak');
sdn2_v = reshape(sdn2,ANx,1);

% %%%%%%%%%%%% K-Wave reconstructions %%%%%%%%%%%%%%%%%%
% 
% create a binary sensor mask of an equivalent continuous circle
sensor_radius_grid_points = round(sensor_radius/dx_rec);
binary_sensor_mask = makeCircle(object_rec.Nx, object_rec.Ny, object_rec.Nx/2, object_rec.Ny/2, sensor_radius_grid_points, sensor_angle);
sensor.mask = binary_sensor_mask;
sdn2_interp = interpCartData(recon_grid, sdn2, cart_sensor_mask, binary_sensor_mask);
IM2_k_interp = inverse(object_rec, time, medium, sensor, sdn2_interp);
% figure;imshow(IM2_k_interp,[]);%title('new image');
Xi1=IM2_k_interp((indxi(1))/2:(indxi(end))/2, (indyi(1))/2:(indyi(end))/2); # Cropp the reconstructed image
Xi = imresize(Xi1,[256,256]);
% figure;imshow(Xi,[]);



save(['/media/fistlab/DATA/Raj/Dataset/mua/0.9/a/X_' num2str(k) '.mat'],'GT','Ximg','Xi');
%imwrite(Img,['/home/fistlab/Raj/Project/Dataset/Image data/X1_' num2str(k) '.JPG']);   
    
end
