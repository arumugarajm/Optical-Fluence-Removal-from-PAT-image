clc;
clear all;

% a = load('C:\Users\arumugaraj\Desktop\d.mat');
% volimage = a.volimage;
load rat_head.mat
% im = imresize3(volimage, [24 24 24]);

[node,elem,face]=vol2mesh(volimage>0.05,1:size(volimage,1),1:size(volimage,2),...
                           1:size(volimage,3),00.5,0.05,1);
                       
                       
%% visualize the resulting mesh

plotmesh(node,face);
axis equal

%% Edited by me
[row,col]=size(node);
% Xim = volimage;
Xim = volimage.*0.03;
% Xim = imresize3(Xim,[256,256,256]);
% Xim = imresize3(Ximo,[24,24,24]);
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
 
 E1(1)=1;E1(2)=2;E1(3)=3;      E2(1)=1;E2(2)=2;E2(3)=3;E2(4)=50;
 F1(1)=1;F1(2)=2;F1(3)=3;F1(4)=4;F1(5)=48;F1(6)=49;F1(7)=50;F1(8)=51;F1(9)=52;F1(10)=53;    F2(1)=1;F2(2)=2;F2(3)=3;F2(4)=4;F2(5)=48;F2(6)=49;F2(7)=50;F2(8)=51;F2(9)=52;F2(10)=53;
 G1(1)=43;G1(2)=44;     G2(1)=42;G2(2)=43;G2(3)=44;
 
  
 
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
%  else
%      Ximn(i) =  Xim(E2(i),F2(i),G2(i));
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

% u=zeros(row,1);
vertn=0.125*[X Y Z];% generating node points according to NIRFAST
% vertn=[X Y Z];


ksi=0.2129*ones(row,1);
c=2.255*10^11*ones(row,1);
kappa=0.33*ones(row,1);
ri=1.33*ones(row,1);
mus=ones(row,1);
num=[1;2;3;4;5;6;7;8;9;10];%10 sources
% num=[1;2;3;4;5;6;7;8;9;10;11;12];%12 sources

coord = 0.125*[34.8536 27.7157 28.4999; 35.0347 23.6106 26.5003; 35.0223 20.4994 24.2973; 35.4907 18.4891 19.4788; 35.2108 16.5002 16.3516; 35.1298 15.4998 12.4634; 35.4316 12.5001 10.9022; 35.489 10.4906 7.79994; 35.4994 10.3152 4.86757; 35.7409 8.50005 2.37469];
% coord = 0.125*0.5*0.5*0.5*[58.9474 255.666 0; 300.255 116.663 0; 461.156 56.6 0; 615.201 19.9372 0; 615.52 12.9839 0; 614.164 27.9428 0; 99.3564 371.897 0; 213.301 379.039 0; 386.415 353.819 0; 612.05 245.915 0];
% coord = [58.9474 255.666 122.409; 300.255 116.663 22.8619; 461.156 56.6 33.9545; 615.201 19.9372 86.5617; 615.52 12.9839 339.946; 614.164 27.9428 501.587; 99.3564 371.897 350.443; 213.301 379.039 521.796; 386.415 353.819 661.683; 612.05 245.915 709.926];
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
% O = ones(row,1);

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
mesh.muan=Ximn;
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

meas.coord = 0.125*[34.8536 27.7157 28.4999; 35.0347 23.6106 26.5003; 35.0223 20.4994 24.2973; 35.4907 18.4891 19.4788; 35.2108 16.5002 16.3516; 35.1298 15.4998 12.4634; 35.4316 12.5001 10.9022; 35.489 10.4906 7.79994; 35.4994 10.3152 4.86757; 35.7409 8.50005 2.37469];
% [58.9474 255.666 0; 300.255 116.663 0; 461.156 56.6 0; 615.201 19.9372 0; 615.52 12.9839 0; 614.164 27.9428 0; 99.3564 371.897 0; 213.301 379.039 0; 386.415 353.819 0; 612.05 245.915 0];
% meas.coord = [58.9474 255.666 122.409; 300.255 116.663 22.8619; 461.156 56.6 33.9545; 615.201 19.9372 86.5617; 615.52 12.9839 339.946; 614.164 27.9428 501.587; 99.3564 371.897 350.443; 213.301 379.039 521.796; 386.415 353.819 661.683; 612.05 245.915 709.926];
[ind, int_func] = mytsearchn(mesh,mesh.source.coord);
mesh.meas=meas;

r=[ind int_func];
mesh.meas.int_func=r;


data = femdata(mesh,0);
% plotmesh(mesh,1);

S = sum(data.phi,2);% add all the fluence generated by the source.  It add all the columns
format long
S;
% min(S);
% S1=zeros(row,1);
S1=full(S);% checking sum and full produce the same result or not


% plotimage(mesh,S1);%plotimage function will plot image for a single variable
new=mesh.mua.*S1;%mua and fluence multiplication
% new=mesh.mua;


% plotmesh1(mesh,new);%plot image for new mua value
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
%  else
%      Ximg(E2(i),F2(i),G2(i))=mesh.mua(i);
end
end

figure;imshow(Xim(:,:,30),[]);
figure;imshow(Ximg(:,:,30),[]);
max(max(max(Ximg)));



addpath('../k_wave_PAT_CODE/k-wave-toolbox-version-1.3/k-Wave');
medium.sound_speed = 1540;  % [m/s]
% create the time array
time.dt = 5e-8;         % sampling time in sec 5e-8
time.length = 2048;      % number of points in time
t2=time.dt;

object_sim.Nx = 201;  % number of grid points in the x (row) direction
object_sim.Ny = 201;  % number of grid points in the y (column) direction
object_sim.Nz = 201;
object_sim.x = 10.1e-3;              % total grid size [m]
object_sim.y = 10.1e-3;              % total grid size [m]
object_sim.z = 10.1e-3;

dx = object_sim.x/(object_sim.Nx-1);              % grid point spacing in the x direction [m]
dy = object_sim.y/(object_sim.Ny-1);              % grid point spacing in the y direction [m]
dz = object_sim.z/(object_sim.Nz-1); 
kgrid = kWaveGrid(object_sim.Nx, dx, object_sim.Ny, dy, object_sim.Nz, dz);

% % define the time array
% kgrid.makeTime(medium.sound_speed);

%%%create a second computation grid for the reconstruction to avoid the inverse crime
object_rec.Nx = (1/2)*(object_sim.Nx-1)+1;    % number of grid points in the x (row) direction
object_rec.Ny = (1/2)*(object_sim.Ny-1)+1;    % number of grid points in the y (column) direction
object_rec.Nz = (1/2)*(object_sim.Nz-1)+1;
object_rec.x = (object_sim.x);              % total grid size [m]
object_rec.y = (object_sim.y);              % total grid size [m]
object_rec.z = (object_sim.z);

dx_rec = (object_rec.x/(object_rec.Nx-1));               % grid point spacing in the x direction [m]
dy_rec = (object_rec.y/(object_rec.Ny-1));              % grid point spacing in the y direction [m]
dz_rec = (object_rec.z/(object_rec.Nz-1));
recon_grid = kWaveGrid(object_rec.Nx, dx_rec, object_rec.Ny, dy_rec, object_rec.Nz, dz_rec);

% define a Cartesian spherical sensor
sensor_radius = 4e-3;       % [m]
center_pos = [0, 0, 0];     % [m]
num_sensor_points = 100;
sensor_mask = makeCartSphere(sensor_radius, num_sensor_points, center_pos, true);

% assign to the sensor structure
sensor.mask = sensor_mask;

M = 25;
N = 26;
P = 22;
indxi = ceil(object_sim.Nx/2) - M:ceil(object_sim.Nx/2) + M -1;
indyi = ceil(object_sim.Ny/2) - N:ceil(object_sim.Ny/2) + N -1;
indzi = ceil(object_sim.Nz/2) - P:ceil(object_sim.Nz/2) + P -1;

Nxi = length(indxi);
Nyi = length(indyi);
Nzi = length(indzi);
tl = time.length;
sml = length(sensor.mask);
ANx = tl*sml;
ANy = Nxi*Nyi*Nzi;



signal_to_noise_ratio = 35;	% [dB]
scaling_factor = 0.15;
IM_init_v = zeros((2*M+1)*(2*N+1)*(2*P+1),1);
p0_magnitude = 1;
% bv=load('D:\project\New folder\acode_fileExchange_941\new codes\reitna-segmentation-master\Ximg.mat','Ximg');
a=imresize3(Ximg,[50,52,44]);
% b=Xim;
% figure;imshow(a,[]);

object_sim.p0 = zeros(object_sim.Nx, object_sim.Ny, object_sim.Nz);
object_sim.p0(indxi,indyi,indzi) = a(:,:,:);



sdn = forward3D(object_sim, time, medium, sensor);

% add noise to the recorded sensor data
sdn2 = addNoise(sdn, signal_to_noise_ratio, 'peak');
sdn2_v = reshape(sdn2,ANx,1);
% figure;imshow(sdn2,[]);

% %%%%%%%%%%%% K-Wave reconstructions %%%%%%%%%%%%%%%%%%
% 
% create a binary sensor mask of an equivalent continuous circle
sensor_radius_grid_points = round(sensor_radius / dx_rec);
binary_sensor_mask = makeSphere(object_rec.Nx, object_rec.Ny,object_rec.Nz, sensor_radius_grid_points);
sensor.mask = binary_sensor_mask;
sdn2_interp = interpCartData(recon_grid, sdn2, sensor_mask, binary_sensor_mask);
IM2_k_interp = inverse3D(object_rec, time, medium, sensor, sdn2_interp);
Xi1=IM2_k_interp((indxi(1))/2:(indxi(end))/2, (indyi(1))/2:(indyi(end))/2, (indzi(1))/2:(indzi(end))/2);
Xi = imresize3(Xi1,[50,52,44]);
figure;imshow(Xi(:,:,30),[]);
figure;imshow(Xim(:,:,30),[]);





