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

function [A, b, B] = SCML_local(xTr, yTr, zTr, numBasis, beta, b_init)

%   SCML-Local
%     xTr: N*D data matrix. Each row is a data point
%     yTr: N*1 label vector. 
%     zTr: N*d embedding matrix. Each row is a kernel PCA embedding of the
%       corresponding training point
%     numBasis: num of basis elements
%     beta: regularization parameter
%     b_init: initialization for b (optional)
%   
%     A, b: parameters of local metrics
%     B: basis set


% generate triplets
T = generate_knntriplets(xTr, yTr, 3, 10);

% generate bases and initialize A and b
B = generate_bases_LDA(xTr, yTr, numBasis);

A = zeros( size(zTr,2), numBasis );

if isempty(b_init)
    b_init = ones(1, numBasis);
end

b = sqrt(b_init');    
b = b + 0.001; % add a small value

% learning A
[A b] = learning_A_stoch_2_1_proximal(A, b, B, xTr, zTr, T, beta);

b = b';
