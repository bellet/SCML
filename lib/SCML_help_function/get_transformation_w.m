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

function L = get_transformation_w(B, w)

% get rid of inactive bases
active_idx = w > 0;
w = w(active_idx);
B = B(active_idx,:);

[K, d] = size(B);

if K < d % if metric is low-rank

    L = zeros(K,d);

    for i = 1:K
        L(i,:) = B(i,:) * sqrt(w(i));
    end

else % if metric is full rank
    
    combM = 0;
    for i = 1:K
        combM = combM + B(i,:)'*B(i,:) * w(i);
    end

    [V,D] = eig(combM);

    % set negative and near-zero eigenvalues to 0
    % to avoid numerical issues
    D = real(diag(D));
    V = real(V);
    j = find(D < 1e-10);
    D(j) = 0;

    L = V * diag(sqrt(D)) * V';

end
