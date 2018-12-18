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

function [xTr, xVa, xTe] = pre_process_data(xTr, xVa, xTe) 

%% normalize data
[dum, xVa]=normalizemeanstd(xTr,xVa);
[xTr, xTe]=normalizemeanstd(xTr,xTe);

%% make the L2 norm of each instance equal to 1
for i=1:700:size(xTr,1)
    BB=min(700,size(xTr,1)-i);
    xTr(i:i+BB,:)=diag((sum(xTr(i:i+BB,:).^2,2)+1e-20).^-0.5)*xTr(i:i+BB,:);
end

for i=1:700:size(xVa,1)
    BB=min(700,size(xVa,1)-i);
    xVa(i:i+BB,:)=diag((sum(xVa(i:i+BB,:).^2,2)+1e-20).^-0.5)*xVa(i:i+BB,:);
end

for i=1:700:size(xTe,1)
    BB=min(700,size(xTe,1)-i);
    xTe(i:i+BB,:)=diag((sum(xTe(i:i+BB,:).^2,2)+1e-20).^-0.5)*xTe(i:i+BB,:);
end