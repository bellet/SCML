function M=mat(C);

r=round(sqrt(size(C,2)));
M=reshape(C,r,r);