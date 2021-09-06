function sensor_data = forward(object, time, medium, sensor)
%FORWARD     Create a data set for PA forward problem.
%
% DESCRIPTION:
%       
%
% USAGE:
%       sensor_data = forward(medium, time, source, sensor)

% INPUTS:
%       medium          -
%       time            -
%       source          - 
%       sensor          -
%
% OUTPUTS:
%       sensor_data     - time varying pressure recorded at the sensor
%                     positions given by sensor.mask
%
% ABOUT:
%       author          - Manojit Pramanik
%       date            - 25th Aug 2012
%       last update     - 25th Aug 2012
%     

dx = object.x/object.Nx;              % grid point spacing in the x direction [m]
dy = object.y/object.Ny;              % grid point spacing in the y direction [m]
kgrid = makeGrid(object.Nx, dx, object.Ny, dy);

kgrid.t_array=0:1:time.length-1;
kgrid.t_array=kgrid.t_array*time.dt;

% smooth the initial pressure distribution and restore the magnitude
p1 = smooth(kgrid, object.p0, true);
source.p0 = p1;

% smooth the source
%source.p = filterTimeSeries(kgrid, medium, source.p);

% set the input options
input_args = {'Smooth', false, 'PMLInside', false, 'PlotPML', false, 'PlotSim', false};
%input_args = {'Smooth', false,'PMLInside',false,'PlotPML',false,'CartInterp', 'nearest', 'PlotSim', false,'DataCast','single'};

% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
