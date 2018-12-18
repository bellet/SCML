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

function [best_A best_b obj n_imp] = learning_A_stoch_2_1_proximal(A, b, B, xTr, zTr, T, beta)

% T: triplet constraints
% SS: target neighbor relationship
% alpha1: regularizer on norm of M (alpha1 is always 0)
% alpha2: regularizer on NN distance term
% beta: regularization parameter on A

dim = size(xTr,2);
N = size(xTr,1);

numAnchor = size(A,2);

dist_diff = compute_dist_diff(T, xTr, B);

MAX_ITER = 200001;      % max number of iterations
stepsize = 0.7;         
output_iter = 5000;     % output every output_iter iterations

%% precomputing matrix dist
[best_A best_b obj n_imp] = opt_procedure_local(A, b, dist_diff, zTr, T, beta, stepsize, MAX_ITER, output_iter);