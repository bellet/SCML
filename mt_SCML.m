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

function [L_set,W,B] = mt_SCML(xTr, yTr, numBasis, beta)

%   mt-SCML
%     xTr: a cell array, the ith cell contains the data matrix 
%       (each row is a point) for ith task 
%     yTr: a cell array, the ith cell contains the label vector for the ith
%       task
%     numBasis: total number of basis elements for all tasks
%     beta: regularization parameter
%
%   L_set: a cell array, the ith cell contains the transformation for each task
%   W: the ith row contains the weight for ith task
%   B: basis set
%   
%   Last Modified: 4/15/2014

n_task = length( xTr );

% generate triplets
for t = 1:n_task
    [T_cell{t}, SS_cell{t}]= generate_knntriplets(xTr{t}, yTr{t}, 3, 10);
end

% generate bases 
B = [];

for t = 1:n_task
    B_part = generate_bases_LDA(xTr{t}, yTr{t}, numBasis/n_task);
    B = [B; B_part];
end

% initializing W
W = ones(n_task, numBasis);

%% learning W
W = learning_W_mt_rda_2_1(W, B, xTr, T_cell, beta); 

for t = 1:n_task
    L_set{t} = get_transformation_w(B, W(t,:));
end