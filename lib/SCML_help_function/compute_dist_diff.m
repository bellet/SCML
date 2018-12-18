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

function dist_diff = compute_dist_diff(T, xTr, B)

% brute force way: compute all pairwise distances
% intractable for large datasets
% dist_old = zeros(size(B,1),size(xTr,1)*size(xTr,1));
% for i = 1:size(B,1)
%     distM = dist_metric( xTr', xTr', reshape( M(i,:),size(B,2), size(B,2)) );
%     dist_old(i,:) = distM(:);
% end

% first compute XTr*B'
xTrB = xTr*B';
% get the list of pairs we actually need to compute
uniqPairs = unique(sort([ T(:,1:2) ; T(:,[1 3]) ]')','rows');
% compute them
dist =  (xTrB(uniqPairs(:,1),:) - xTrB(uniqPairs(:,2),:)).^2;
% build an index to reference them
index = sparse(uniqPairs(:,1),uniqPairs(:,2),[1:size(uniqPairs,1)]');

% build the dist difference for all triplets
dist_diff = zeros(size(T,1),size(B,1));
for idx = 1:size(T,1)
    dist_diff(idx,:) = dist(index(min(T(idx,1),T(idx,2)),max(T(idx,1),T(idx,2))),:) - dist(index(min(T(idx,1),T(idx,3)),max(T(idx,1),T(idx,3))),:);
end

