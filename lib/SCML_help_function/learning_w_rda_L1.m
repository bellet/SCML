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

function [best_w obj n_imp] = learning_w_rda_L1(w, B, xTr, T, beta)

% T: triplet constraints
% beta: regularization parameter on w

gamma = 5e-3;  % note: the bigger gamma, the smaller the stepsize

dist_diff = compute_dist_diff(T, xTr, B);

%% precomputing matrix dist 
best_obj = 1e10;

w = zeros(1,size(w,2));
avg_grad_w = zeros(1,size(w,2));
sizeT = size(T,1);

MAX_ITER = 100001;      % max number of iterations
output_iter = 5000;     % output every output_iter iterations

[best_w, obj, n_imp] = opt_procedure_global(w, avg_grad_w, dist_diff, sizeT, beta, gamma, MAX_ITER, output_iter);