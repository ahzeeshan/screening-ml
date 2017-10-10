%This code will take the numner of neurons and do the calcs
%%
clear all
%close all
%clc
						    tic;

mycluster = parcluster('local')
  mycluster.JobStorageLocation = strcat(getenv('SCRATCH'),'/.matlab/', getenv('SLURM_JOB_ID'))
  mycluster.NumWorkers = str2num(getenv('SLURM_JOB_CPUS_PER_NODE'))
  parpool(mycluster, mycluster.NumWorkers)
  %parpool(mycluster, mycluster.NumWorkers)
  saveProfile(mycluster)

  lattice = fileread('lattice-type.txt');
load(strcat(lattice,'_results.mat'))
load(fullfile('..','data-gen',strcat(lattice,'-non-training-data.mat')))

load(fullfile('..','data-gen',strcat(lattice,'-data.mat'))) % xdata and ydata
load(fullfile('..','Linear',strcat('features_',lattice,'.mat')))

%% User input data
X_mat = xdata(1:end-floor(0.1*size(xdata,1)),:);
coeffs = ydata(1:end-floor(0.1*size(xdata,1)),:);
num_coeffs = size(coeffs,2);
cubic_nt = xntdata;

net_storage_complete = cell(num_coeffs,1);
tr_storage_complete = cell(num_coeffs,1);
test_err_complete = cell(num_coeffs,1);
index_out_coeffs = cell(num_coeffs,1);

% cubic_nt = cubic_nt';  % Just converting the array to transpose %Now this becomes 692X13
for coeff_num = 1:1:num_coeffs
    coeff_num
    hidden_layer_size = hidden_layer_av(coeff_num) %you can provide a row vector which can represent if we need multiple hidden layers
    % pre processing normalization
    feature_arr = feature_list{coeff_num};
    
    X1 = X_mat(:,feature_arr);
    non_training_yy = cubic_nt(:,feature_arr)';
    
    yy = X1'; %Doing transpose to feed into Neural network
    
    %Mapminmax starts below to map x from [-1,1]
    
    [x, tot_inp_recover] = mapminmax(yy);
    % tot_inp is our favorable output and tot_inp_recover is the structure by
    % which we can apply that transformation to other values as well and can
    % also get the original value of tot_inp.
    non_training = mapminmax('apply',non_training_yy,tot_inp_recover);
    
    t_yy = coeffs(:,coeff_num)'; % t here refers to the target in ANNs
    [t, t_recover] = mapminmax(t_yy); %Mapping the coefficients as well in [-1,1]
    Y = t';
    %% Looping
    sample_test = 1000;
    trainFcn = 'trainlm';
    XTRAIN = x;
    ytrain = t;
    %XTEST = XTEST';

val_perf_reqd=0.01;
tr_perf_reqd = 0.01;
reg_tr_reqd=0.9;
Rsq_val_reqd = 0.65;
Rsq_tr_reqd = 0.8; 
max_index = 1000;
ind_out = zeros(sample_test, 1);
net_storage = cell(sample_test,1);
tr_storage = cell(sample_test,1);
    
parfor random_weights = 1:1:sample_test
        random_weights
  val_min = 1;
tr_perf_min = 1;
net_min = [];
bool_var = 1;
index_watch = 1;
        
        
net = fitnet(hidden_layer_size,trainFcn);
net = configure(net,XTRAIN,ytrain);
net.trainParam.showWindow = 0;
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100;
%        net.performParam.regularization = 0.01;
net.trainParam.max_fail = 15;
        
while((bool_var))
  init(net);
[net,tr] = train(net,XTRAIN,ytrain);
out_train = net(XTRAIN);
trTarg = ytrain(tr.trainInd);
trOut = out_train(tr.trainInd);
[reg_tr,~,~] = regression(trTarg, trOut);
Rsq_tr = 1 - sum((ytrain(tr.trainInd) - out_train(tr.trainInd)).^2)/sum((ytrain(tr.trainInd) - mean(ytrain(tr.trainInd))).^2);
Rsq_val = 1 - sum((ytrain(tr.valInd) - out_train(tr.valInd)).^2)/sum((ytrain(tr.valInd) - mean(ytrain(tr.valInd))).^2);
%[reg_val,~,~] = regression(vTarg, vOut);
%[reg_test,~,~] = regression(testTarg, testOut)
  val_perf = mse(net,ytrain(tr.valInd),out_train(tr.valInd));
tr_perf = mse(net,ytrain(tr.trainInd),out_train(tr.trainInd));
%bool_var = (val_perf>val_perf_reqd)||(reg_tr<reg_tr_reqd); % (tr_perf>tr_perf_reqd) Training parameter
bool_var = (Rsq_val < Rsq_val_reqd) || (Rsq_tr < Rsq_tr_reqd)
            if val_perf<val_min %&& tr_perf<tr_perf_min
  val_min = val_perf;
tr_perf_min = tr_perf;
net_min = net;
tr_min = tr;
            end
            if index_watch>=max_index % this is just for the case when the iterations do not converge even after 1000 runs ....
											   ind_out(random_weights) = 1;
break;
            end
            ind_out(random_weights) = 0; % Flag for whether it found within 1000 iters 0<->yes
            index_watch = index_watch +1;
            
        end
        net_storage{random_weights} = net_min;
tr_storage{random_weights} = tr_min;
        
    end
    net_storage_complete{coeff_num} = net_storage;
tr_storage_complete{coeff_num} = tr_storage;
index_out_coeffs{coeff_num} = ind_out;
%test_err_complete{coeff_num} = test_err;
end
toc;
save(strcat(lattice,'_final_results.mat'),'net_storage_complete','index_out_coeffs','tr_storage_complete')
