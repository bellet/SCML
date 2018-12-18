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

function dist = dist_metric(X, Y, M)

% X: d*n
% Y: d*m
% M: d*d

% dist: n*m

n = size(X,2);
m = size(Y,2);

dist = repmat( diag(X'*M*X), 1, m) - 2 * X'*M*Y + repmat( diag(Y'*M*Y)', n, 1 );
