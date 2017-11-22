clear all
%close all
%clc
tic;

% mycluster = parcluster('local')
%   mycluster.JobStorageLocation = strcat(getenv('SCRATCH'),'/.matlab/', getenv('SLURM_JOB_ID'))
%   mycluster.NumWorkers = str2num(getenv('SLURM_JOB_CPUS_PER_NODE'))
%   parpool(mycluster, mycluster.NumWorkers)
%   %parpool(mycluster, mycluster.NumWorkers)
%   saveProfile(mycluster)

lattice = strtrim(fileread('lattice-type.txt'));
disp(lattice)
load(fullfile('..','data-gen',strcat(lattice,'-data-posd.mat'))) % xdata and ydata
%load(fullfile('..','Linear',strcat('features_',lattice,'.mat')))
%load(fullfile('..','data-gen',strcat(lattice,'-non-training-data.mat')))
X_mat = xdata(1:end-floor(0.1*size(xdata,1)),:);

%% Here is where you need to change the things for different coefficients
coeffs = ydata(1:end-floor(0.1*size(xdata,1)),:);
num_coeffs = size(coeffs,2);
%cubic_nt = xntdata; % 626 by 18
num_kfolds = 2;

hidden_layer_min = zeros(1,num_kfolds);
sample_test = 2;
max_index = 1000;
Rsq_val_reqd = -inf; val_perf_reqd = 0.01;
Rsq_tr_reqd = -inf; reg_tr_reqd = 0.85;
tr_perf_reqd = 0.01;


%% Loading the data and normalising it to [-1 ,1]
X1 = X_mat(:,:);
yy = X1'; %Doing transpose to feed into Neural network
%Mapminmax starts below to map x from [-1,1], row min and max
[x, tot_inp_recover] = mapminmax(yy);
[pcacoeff,score,latent,tsquared,explained,mu] = pca(x');
pcomps = find(latent>0.2);
x = score(:,pcomps);
% tot_inp is our favorable output and tot_inp_recover is the structure by
% which we can apply that transformation to other values as well and can
% also get the original value of tot_inp.
%non_training = mapminmax('apply',non_training_yy,tot_inp_recover);

t_yy = coeffs(:,:)'; % t here refers to the target in ANNs

[t, t_recover] = mapminmax(t_yy); %Mapping the coefficients as well in [-1,1]
%Y = t';
k = 10;
%max_neuron(coeff_num) = floor( (size(X_mat,1)-1)/(length(feature_list{coeff_num})+2));
max_neuron = length(pcomps);
mse_size = zeros(num_kfolds,max_neuron);
%mmax = max_neuron(coeff_num);

if floor(max_neuron)==1
    for random_kfold=1:num_kfolds
        hidden_layer_min(random_kfold) = 1;
        mse_size(random_kfold,:) = -1;
    end
else
    for random_kfold=1:num_kfolds
        random_kfold
        % running iterations for hidden layer size
        mse_temp_val = zeros(1,max_neuron);
        for layer_size = 1:max_neuron
            layer_size
            fun = @(XTRAIN,ytrain,XTEST,ytest) nntrain(XTRAIN,ytrain,XTEST,ytest,layer_size,sample_test,max_index,Rsq_val_reqd, Rsq_tr_reqd);
              mse_temp_val(layer_size) = sum(crossval(fun,x,t','kfold',k));
        end
        mse_size(random_kfold,:) = mse_temp_val;
        [err,ind_err] = min(mse_temp_val);
        hidden_layer_min(random_kfold) = ind_err;
    end
end

mse_fold_av = mean(mse_size);
[~,hidden_layer_av] = min(mse_fold_av);
max_neuron
hidden_layer_av
%mse_fold_av = mean(mse_size,2);
%mse_fold_av = reshape(mse_fold_av, [num_coeffs, length(hidden_layer_size_range)]);

%[mse_min,ind_min_av] = min(mse_fold_av,[],2);
%hidden_layer_av = ind_min_av;

save(strcat(lattice,'_results_pca.mat'),'max_neuron','mse_size','hidden_layer_min','mse_fold_av','hidden_layer_av');
toc;
