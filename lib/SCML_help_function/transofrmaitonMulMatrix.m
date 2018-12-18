% Copyright 2014 Yuan Shi
% yuanshi@usc.edu
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

function Lout=transofrmaitonMulMatrix(M)
Lout=zeros(size(M));
for i=1:size(M,1)
    Q=mat(M(i,:));
[L,dd]=eig(Q);
dd=real(diag(dd));
L=real(L);
j=find(dd<1e-10);
dd(j)=0;
L=(L*diag(sqrt(dd)))';
Lout(i,:)=vec(L)';
end
end