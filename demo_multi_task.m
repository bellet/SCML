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

%   demo of st-SCML and mt-SCML

setpaths;

%% initialize random number generator
init_rng();

%% load data
load('dataset/sentiment_mt.mat');

% preprocessing
n_task = length(xTr);   


for t = 1:n_task
    [xTr{t}, xVa{t}, xTe{t}] = pre_process_data(xTr{t}, xVa{t}, xTe{t});
end

%% single-task learning
for t = 1:n_task
    st_L_set{t} = SCML_global(xTr{t}, yTr{t}, 400, 1e-4);
end

for t = 1:n_task
    err_st{t} = knnclassifytree(st_L_set{t},xTr{t}',yTr{t}',xTe{t}',yTe{t}',3);
end

mt_L_set = mt_SCML(xTr, yTr, 100*n_task, 1e-4);

for t = 1:n_task
    err_mt{t} = knnclassifytree(mt_L_set{t},xTr{t}',yTr{t}',xTe{t}',yTe{t}',3); 
end

fprintf('Single-task learning\n');
res_train = zeros(n_task,1);
res_test = zeros(n_task,1);
for t = 1:n_task
    fprintf('%g task: train %.2f\t test %.2f\n', t, err_st{t}(1)*100, err_st{t}(2)*100);
    res_train(t) = err_st{t}(1)*100;
    res_test(t) = err_st{t}(2)*100;
end
fprintf('Average: train %.2f\t test %.2f\n\n', mean(res_train), mean(res_test));

fprintf('Multi-task learning\n');
for t = 1:n_task
    fprintf('%g task: train %.2f\t test %.2f\n', t, err_mt{t}(1)*100, err_mt{t}(2)*100);
    res_train(t) = err_mt{t}(1)*100;
    res_test(t) = err_mt{t}(2)*100;    
end
fprintf('Average: train %.2f\t test %.2f\n', mean(res_train), mean(res_test));