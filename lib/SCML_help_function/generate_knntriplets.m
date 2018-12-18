
function [triplets,pairset]= generate_knntriplets(Xtr, Ytr, knn_par1, knn_par2)

% knn_par1: #target neighbors
% knn_par2: #imposters

fprintf('Creating %d-NN triplets\n',knn_par1);

[nos,nof]=size(Xtr);

triplets=zeros(nos*knn_par1*knn_par2,3);

UY=unique(Ytr);

diff_index=zeros(knn_par2,nos);
for cc = 1:length(UY)
    fprintf('%i nearest imposture neighbors for class %i :\n',knn_par2,UY(cc));
    i=find(Ytr==UY(cc));
    j=find(Ytr~=UY(cc));
    nn=LSKnn(Xtr(j,:)',Xtr(i,:)', 1:knn_par2);
    diff_index(:,i)=j(nn);
end

same_index=zeros(knn_par1,nos);

for cc = 1:length(UY)
    fprintf('%i nearest genuine neighbors for class %i:\n',knn_par1,UY(cc));
    i=find(Ytr== UY(cc));
    class_size = size(i,1);
    nn=LSKnn(Xtr(i,:)',Xtr(i,:)', 2:knn_par1+1);
    % if class_size too small to find enough genuine neighbors
    % we keep the elements to zero and delete them later
    same_index(1:min(knn_par1,class_size-1),i)=i(nn(1:min(knn_par1,class_size-1),:));
end

clear i j nn;
triplets(:,1)=vec(repmat([1:nos],knn_par1*knn_par2,1));
temp=zeros(knn_par1*knn_par2,nos);
for i=1:knn_par1
    temp((i-1)*knn_par2+1:i*knn_par2,:)=repmat(same_index(i,:),knn_par2,1);
end
triplets(:,2)=vec(temp);


triplets(:,3)=vec(repmat(diff_index,knn_par1,1));



pairset=zeros(2,nos*knn_par1);
pairset(1,:)=vec(repmat([1:nos],knn_par1,1));
pairset(2,:)=vec(same_index);

% remove missing triplets / pairs (0 valued)
triplets = triplets(all(triplets,2),:);
pairset = pairset(:,all(pairset,1));

fprintf('totally %d triplets for training\n\n', size(triplets,1));
