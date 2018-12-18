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

function [L,w,B] = SCML_global(xTr, yTr, numBasis, beta)

%   SCML-Global
%     xTr: N*D data matrix. Each row is a data point
%     yTr: N*1 label vector. 
%     numBasis: num of basis elements
%     beta: regularization parameter
%
%     L: global transformation. The corresponding metric is L'*L.
%     w: global weight
%     B: basis set
%

% generating triplets
% using 3 target neighbors and 10 imposters
T = generate_knntriplets(xTr, yTr, 3, 10);

% generating LDA basis
B = generate_bases_LDA(xTr, yTr, numBasis);

% initializing w
w = ones(1, numBasis);

% learning w
[w obj n_imp] = learning_w_rda_L1(w, B, xTr, T, beta);

L = get_transformation_w(B, w);