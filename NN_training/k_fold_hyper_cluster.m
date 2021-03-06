clear all
%close all
%clc
tic;

mycluster = parcluster('local')
mycluster.JobStorageLocation = strcat('/home/azeeshan/.matlab','/', getenv('SLURM_JOB_ID'))
parpool(mycluster, str2num(getenv('SLURM_CPUS_ON_NODE')))
mycluster.NumWorkers = 24
saveProfile(mycluster)

lattice = strtrim(fileread('lattice-type.txt'));
disp(lattice)
load(fullfile('..','data-gen',strcat(lattice,'-data.mat'))) % xdata and ydata
load(fullfile('..','Linear',strcat('features_',lattice,'.mat')))
load(fullfile('..','data-gen',strcat(lattice,'-non-training-data.mat')))
X_mat = xdata(1:end-floor(0.1*size(xdata,1)),:);

%% Here is where you need to change the things for different coefficients
coeffs = ydata(1:end-floor(0.1*size(xdata,1)),:);
num_coeffs = size(coeffs,2);
cubic_nt = xntdata; % 626 by 18
hidden_layer_size_range = (1:10);
num_kfolds = 5;
mse_size = zeros(num_coeffs, num_kfolds, length(hidden_layer_size_range));
hidden_layer_min = zeros(num_coeffs,num_kfolds);
sample_test = 20;
max_index = 1000;
val_perf_reqd = 0.01;
reg_tr_reqd = 0.85;
tr_perf_reqd = 0.01;
for coeff_num = 1:1:num_coeffs % for all the coefficients
    %% Loading the data and normalising it to [-1 ,1]
    coeff_num
    feature_arr = feature_list{coeff_num};
    X1 = X_mat(:,feature_arr);
    non_training_yy = cubic_nt(:,feature_arr)';
    
    yy = X1'; %Doing transpose to feed into Neural network
    
    %Mapminmax starts below to map x from [-1,1], row min and max
    [x, tot_inp_recover] = mapminmax(yy);
    % tot_inp is our favorable output and tot_inp_recover is the structure by
    % which we can apply that transformation to other values as well and can
    % also get the original value of tot_inp.
    non_training = mapminmax('apply',non_training_yy,tot_inp_recover);
    
    t_yy = coeffs(:,coeff_num)'; % t here refers to the target in ANNs
    [t, t_recover] = mapminmax(t_yy); %Mapping the coefficients as well in [-1,1]
    Y = t';
    
    k = 5;
    
    max_neuron = size(X_mat,1)/(2*length(feature_list{coeff_num}));
    
    if floor(max_neuron)==1
        for random_kfold=1:num_kfolds
            hidden_layer_min(coeff_num,random_kfold) = 1;
        end
    else
        hidden_layer_size_range = 1:max_neuron;
        for random_kfold=1:num_kfolds
            random_kfold
            % running iterations for hidden layer size
            parfor layer_size = hidden_layer_size_range
                layer_size
                fun = @(XTRAIN,ytrain,XTEST) neural_net(XTRAIN,ytrain,XTEST,layer_size,sample_test,max_index,val_perf_reqd, tr_perf_reqd);
                mse_size(coeff_num, random_kfold, layer_size) = crossval('mse',x',t','Predfun',fun,'kfold',k);
            end
            [err,ind_err] = min(mse_size(coeff_num,random_kfold,:));
            hidden_layer_min(coeff_num,random_kfold) = ind_err;
        end
    end
end

mse_fold_av = mean(mse_size,2);
mse_fold_av = reshape(mse_fold_av, [num_coeffs, length(hidden_layer_size_range)]);

[mse_min,ind_min_av] = min(mse_fold_av,[],2);
hidden_layer_av = ind_min_av;

save(strcat(lattice,'_results.mat'),'mse_size','hidden_layer_min','mse_fold_av','hidden_layer_av');
toc;
