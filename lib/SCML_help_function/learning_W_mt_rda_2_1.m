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

function [best_W obj n_imp] = learning_W_mt_rda_2_1(W, B, xTr, T_cell, beta)

% T_cell: triplet constraints
% beta: regularization parameter on W

MAX_ITER = 100000;
n_task = length( T_cell );

for t = 1:n_task
    dist_diff_cell{t} = compute_dist_diff(T_cell{t}, xTr{t}, B);
end

W = zeros(n_task,size(W,2));
avg_grad_W = zeros(n_task,size(W,2));
norm_col = zeros(size(W,2)); % for book-keeping of norms of colums

% set gamma here
% note: the bigger gamma, the smaller the stepsize
% in this case, only used for beta == 0 (no reg)
gamma = 1e-3;

sizeT_set = zeros(n_task,1);
for t = 1:n_task
    sizeT_set(t) = size(T_cell{t},1);
end

MAX_ITER = 60001;
output_iter = 5000;

[best_W, obj, n_imp] = opt_procedure_mt(W, avg_grad_W, dist_diff_cell, norm_col, sizeT_set, beta, gamma, MAX_ITER, output_iter);
