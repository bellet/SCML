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

function B = generate_bases_LDA(xTr, yTr, numBasis)

% generate basis from local LDA

fprintf('Generating LDA bases...\n');

UY = unique(yTr);
n_class = length(unique(yTr));

num_eig = min( n_class - 1, size(xTr,2) );

n_cluster = ceil( numBasis/(2 * num_eig) );

rand('seed', 2013);
[clus cX] = kmeans(xTr, n_cluster,'Replicates',5,'EmptyAction','singleton');

dim = size(xTr, 2);

if dim > 50
    nK = 50;
else
    nK = 10;
end

class_count = hist(yTr,UY); % count number of points per class
% number of points taken in each class is the minimum between the class size and nK
nK_class = min(class_count,nK);

idx_set = zeros( n_cluster, sum(nK_class) );

for c = 1:n_class
    sel_c = find( yTr ==  UY(c) );
    dist = L2_distance(cX', xTr(sel_c,:)');
    [vals, id] = sort(dist,2);
    idx_set(:, sum(nK_class(1:c-1))+1:sum(nK_class(1:c)) ) = sel_c( id(:,1:nK_class(c)) );
end

B = zeros(numBasis,size(xTr,2));

for i = 1:n_cluster
    [eigvector, eigvalue] = LDA( yTr( idx_set(i,:) ), [], xTr(idx_set(i,:),:) );
    B( num_eig*(i-1)+1: num_eig*i, :) = eigvector';
end

nK = 20;

nK_class = min(class_count,nK);

idx_set = zeros( n_cluster, sum(nK_class) );

for c = 1:n_class
    sel_c = find( yTr ==  UY(c) );
    dist = L2_distance(cX', xTr(sel_c,:)');
    [vals, id] = sort(dist,2);
    idx_set(:, sum(nK_class(1:c-1))+1:sum(nK_class(1:c)) ) = sel_c( id(:,1:nK_class(c)) );
end

sta_pos = num_eig * n_cluster;
for i = 1:n_cluster
    [eigvector, eigvalue] = LDA( yTr( idx_set(i,:) ), [], xTr(idx_set(i,:),:) );
    B( sta_pos+num_eig*(i-1)+1: sta_pos+num_eig*i, :) = eigvector';
end

extra_num = size(B,1)-numBasis;
fprintf('removing %g basis\n', extra_num);
B(numBasis+1:end,:) = [];