function [U,Xd,Yd,Zd] = kpca(X,sigma,d,Y,Z)
% Guassian kernel principal component analysis.
%
% Input:
% 	X: NxD matrix containing training points rowwise
% 	sigma: Gaussian kernel width
%	d: latent dimension
%	Y: MxD test set
%
% Output:
% 	U: eigenvectors of kernel matrix, normalized according to eigenvalues
% 	Xd: latent representation for training set
% 	Yd: latent representation for test set
%
% Copyright (c) 2011 by Weiran Wang. 
% If you find any problem, please contact me at wwang5@ucmerced.edu.

[N,D] = size(X);
G = exp(-sqdist(X)/sigma);	% Kernel Gram matrix.
C = eye(N)-ones(N)/N;			% Centering matrix.

opts.disp=0;
[U,L] = eigs(C*G*C,d,'LM',opts);
L = diag(L);
U = U * diag( sparse( 1./sqrt(L(1:d)) ) );
Xd = U * diag( sparse(L(1:d)) );

%MsERR=triu(ones(N,N),1)*L/N;
%MsERR(d)

%Residual =(sum(diag(C*G*C))-sum(L(1:d)))/N;

[M,D] = size(Y);
T = exp(-sqdist(Y,X)/sigma);
Yd = (T - (ones(M,N)/N*G))*C*U;

[O,D] = size(Z);
V = exp(-sqdist(Z,X)/sigma);
Zd = (V - (ones(O,N)/N*G))*C*U;


% sqd = sqdist(X[,Y,w]) Matrix of squared (weighted) Euclidean distances d(X,Y)
%
% If X and Y are matrices of row vectors of the same dimension (but possibly
% different number of rows), then sqd is a symmetric matrix containing the
% squared (weighted) Euclidean distances between every row vector in X and
% every row vector in Y.
% The square weighted Euclidean distance between two vectors x and y is:
%   sqd = \sum_d { w_d (x_d - y_d)Â² }.
%
% sqdist requires memory storage for around two matrices of NxM.
%
% NOTE: while this way of computing the distances is fast (because it is
% vectorised), it is slightly inaccurate due to cancellation error, in that
% points Xn and Ym closer than sqrt(eps) will have distance zero. Basically,
% when |a-b| < sqrt(eps) (approx. 1.5e-8) then aÂ²+bÂ²-2ab becomes numerically
% zero while (a-b)Â² is nonzero.
%
% In:
%   X: NxD matrix of N row D-dimensional vectors.
%   Y: MxD matrix of M row D-dimensional vectors. Default: equal to X.
%   w: 1xD vector of real numbers containing the weights (default: ones).
% Out:
%   sqd: NxM matrix of squared (weighted) Euclidean distances. sqd(n,m)
%      is the squared (weighted) Euclidean distance between row vectors
%      X(n,:) and Y(m,:).
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2009 by Miguel A. Carreira-Perpinan

function sqd = sqdist(X,Y,w)

if nargin==1	% Fast version for common case sqdist(X)
  x = sum(X.^2,2); sqd = max(bsxfun(@plus,x,bsxfun(@plus,x',-2*X*X')),0);
  return
end

% ---------- Argument defaults ----------
if ~exist('Y','var') | isempty(Y) Y = X; eqXY = 1; else eqXY=0; end;
% ---------- End of "argument defaults" ----------
  
if exist('w','var') & ~isempty(w)
  h = sqrt(w(:)'); X = bsxfun(@times,X,h);
  if eqXY==1 Y = X; else Y = bsxfun(@times,Y,h); end;
end

% We ensure that no value is negative (which can happen due to precision loss
% when two vectors are very close).
x = sum(X.^2,2);
if eqXY==1 y = x'; else y = sum(Y.^2,2)'; end;
sqd = max(bsxfun(@plus,x,bsxfun(@plus,y,-2*X*Y')),0);