function [Eval,Details]=knnclassify_local(B,xTr,lTr,W,xTe,lTe,Wtest,KK,varargin)
% function [Eval,Details]=knnclassify(L,xTr,yTr,xTe,yTe,Kg);
%
% INPUT:
%  B	:   basis set
%  xTr	:   training vectors (each column is an instance)
%  yTr	:   training labels  (row vector!!)
%  W  :   basis weights for training data
%  xTe  :   test vectors
%  yTe  :   test labels
%  WTest  :   basis weights for test data
%  KK	:   number of nearest neighbors
%


pars.train=1;
pars.test=1;
pars.cosigndist=0;
pars.blocksize=700;
pars=extractpars(varargin,pars);

MM=min([lTr lTe]);
if(nargin<7)
    KK=3;
end;

if(length(KK)==1) outputK=ceil(KK/2);KK=1:2:KK;else outputK=1:length(KK);end;

Kn=max(KK);




[NTr]=size(xTr,2);
[NTe]=size(xTe,2);

Eval=zeros(2,length(KK));
lTr2=zeros(length(KK),NTr);
lTe2=zeros(length(KK),NTe);

iTr=zeros(Kn,NTr);
iTe=zeros(Kn,NTe);




if(~pars.train)
    NTr=0;
end;

xTrNew = B*xTr;
sx1=xTrNew.^2;
for i=1:pars.blocksize:max(NTr)
    if(pars.train && i<=NTr)
        BTr=min(pars.blocksize-1,NTr-i);
        Dtr=zeros(size(xTr,2),BTr+1);
        for j=1:size(B,1)
            Dtr=Dtr+repmat(W(i:i+BTr,j)',size(xTrNew,2),1).*bsxfun(@plus,bsxfun(@plus,-2*xTrNew(j,:)'*xTrNew(j,i:i+BTr),sx1(j,:)'),sx1(j,i:i+BTr));
        end
        
        [dist,nn]=mink(Dtr,Kn+1);
        nn=nn(2:Kn+1,:);
        lTr2(:,i:i+BTr)=LSKnn2(lTr(nn),KK,MM);
        iTr(:,i:i+BTr)=nn;
        Eval(1,:)=sum((lTr2(:,1:i+BTr)~=repmat(lTr(1:i+BTr),length(KK),1))',1)./(i+BTr);
    end;
end

xTeNew = B*xTe;
sx2=xTeNew.^2;
for i=1:pars.blocksize:max(NTe)
    if(pars.test && i<=NTe)
        BTe=min(pars.blocksize-1,NTe-i);
        Dtr=zeros(size(xTr,2),BTe+1);
        for j=1:size(B,1)
            Dtr=Dtr+repmat(Wtest(i:i+BTe,j)',size(xTrNew,2),1).*bsxfun(@plus,bsxfun(@plus,-2*xTrNew(j,:)'*xTeNew(j,i:i+BTe),sx1(j,:)'),sx2(j,i:i+BTe));
        end
        [dist,nn]=mink(Dtr,Kn);
        lTe2(:,i:i+BTe)=LSKnn2(reshape(lTr(nn),max(KK),BTe+1),KK,MM);
        iTe(:,i:i+BTe)=nn;
        Eval(2,:)=sum((lTe2(:,1:i+BTe)~=repmat(lTe(1:i+BTe),length(KK),1))',1)./(i+BTe);
    end;
end;


% create "Details" output
if(pars.test)
    Details.lTe2=lTe2;
    Details.iTe=iTe;
end;
if(pars.train)
    Details.lTr2=lTr2;
    Details.iTr=iTr;
end;

% extract "Eval" output
if(pars.train && pars.test)
    Eval=Eval(:,outputK);
end;
if(pars.train && ~pars.test)
    Eval=Eval(1,outputK);
end;
if(~pars.train && pars.test)
    Eval=Eval(2,outputK);
end;

function yy=LSKnn2(Ni,KK,MM)
% function yy=LSKnn2(Ni,KK,MM);
%

if(nargin<2)
    KK=1:2:3;
end;

N=size(Ni,2);
Ni=Ni-MM+1;
classes=unique(unique(Ni))';

%yy=zeros(1,size(Ni,2));
%for i=1:size(Ni,2)
%  n=zeros(max(un),1);
%  for j=1:size(Ni,1)
%     n(Ni(j,i))=n(Ni(j,i))+1;
%  end;
%  [temp,yy(i)]=max(n);
%end;



T=zeros(length(classes),N,length(KK));


for i=1:length(classes)
    c=classes(i);
    for k=KK
        %  NNi=Ni(1:k,:)==c;
        %  NNi=NNi+(Ni(1,:)==c).*0.01;% give first neighbor tiny advantage
        try
            T(i,:,k)=sum(Ni(1:k,:)==c,1);
        catch
            keyboard;
        end;
    end;
end;

yy=zeros(max(KK),N);
for k=KK
    [temp,yy(k,1:N)]=max(T(:,:,k)+T(:,:,1).*0.01);
    yy(k,1:N)=classes(yy(k,:));
end;
yy=yy(KK,:);

yy=yy+MM-1;




function dist=spdistance(X,x,X2,x2,pars);
[D,N] = size(X);
[d,n] = size(x);
%X2 = sum(X.^2,1);
%x2 = sum(x.^2,1);

% PAIRWISE DISTANCES
if(~pars.cosigndist)	% sparse L2 distance
    if(D~=d),error('Both sets of vectors must have same dimensionality!\n');end;
    dist = full(repmat(x2,N,1)+repmat(X2.',1,n)-2*X.'*x);
else
    dist=full(X'*x);	% sparse cosign similarity
    dist=dist./repmat(sqrt(x2),N,1)./repmat(sqrt(X2'),1,n);
    dist=1-dist;
end;



