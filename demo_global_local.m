% Copyright 2014 Yuan Shi & Aurelien Bellet
% 
% This file is part of SCML.
% 
% SCML is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% SCML is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with SCML.  If not, see <http://www.gnu.org/licenses/>.

%   demo of SCML-Global and SCML-Local

setpaths;

%% initialize random number generator
init_rng();

%% load data
load('dataset/vehicle.mat');

[xTr, xVa, xTe] = pre_process_data(xTr, xVa, xTe);

dim = size(xTr,2);

%% Euclidean metric
err_I = knnclassifytree(eye(dim),xTr',yTr',xTe',yTe',3);


%% SCML-Global
[L, w_global] = SCML_global(xTr, yTr, 400, 1e-5);
err_global = knnclassifytree(L,xTr',yTr',xTe',yTe',3);


%% SCML-Local
[zTr, zVa, zTe] = get_kpca_embedding(xTr, yTr, xVa, xTe, 40);

[A, b, B] = SCML_local(xTr, yTr, zTr, 400, 1e-5, w_global);
% classification
W = (zTr * A + repmat( b', size(zTr,1), 1) ).^2;
Wtest = (zTe * A + repmat( b', size(zTe,1), 1) ).^2;
err_local = knnclassify_local(B, xTr', yTr', W, xTe', yTe', Wtest, 3);

fprintf('Euclidean error: train %.2f  test %.2f \n', 100*err_I(1), 100*err_I(2));
fprintf('SCML-Global error: train %.2f  test %.2f \n', 100*err_global(1), 100*err_global(2));
fprintf('SCML-Local error: train %.2f  test %.2f \n', 100*err_local(1), 100*err_local(2));