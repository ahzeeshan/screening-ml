%This code will take the numner of neurons and do the calcs
%%
clear all
close all
clc
tic;
lattice = 'cubic';
load(strcat(lattice,'_results.mat'))
load(fullfile('..','data-gen',strcat(lattice,'-non-training-data.mat')))

load(fullfile('..','data-gen',strcat(lattice,'-data.mat'))) % xdata and ydata
load(fullfile('..','Linear',strcat('features_',lattice,'.mat')))

% Its output is feature_list
%% User input data
X_mat = xdata;
coeffs = ydata;
num_coeffs = size(coeffs,2);
cubic_nt = xntdata;

net_storage_complete = cell(num_coeffs,1);
test_err_complete = cell(num_coeffs,1);
index_out_coeffs = cell(num_coeffs,1);
% cubic_nt = cubic_nt';  % Just converting the array to transpose %Now this becomes 692X13
parfor coeff_num = 1:1:num_coeffs
    hidden_layer_size = hidden_layer_av(coeff_num); % you can provide a row vector which can represent if we need multiple hidden layers
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
    val_min = 1;
    tr_perf_min = 1;
    net_min = [];
    val_perf_reqd=0.05;
    reg_tr_reqd=0.85;
    max_index = 1000;
    for random_weights = 1:1:sample_test
        net = fitnet(hidden_layer_size,trainFcn);
        net = configure(net,XTRAIN,ytrain);
        net.trainParam.showWindow = 0;
        net.divideParam.trainRatio = 80/100;
        net.divideParam.valRatio = 20/100;
        net.divideParam.testRatio = 0/100;
        net.performParam.regularization = 0;
        bool_var = 1;
        index_watch = 1;
        %net_min = [];
        while((bool_var))
            init(net);
            [net,tr] = train(net,XTRAIN,ytrain);
            out_train = net(XTRAIN);
            %e = gsubtract(XTRAIN,ytrain);
            %performance = perform(net,ytrain,out_train);
            if(size(ytrain) ~= size(out_train))
                ytrain = ytrain';
            end
            trTarg = ytrain(tr.trainInd);
            trOut = out_train(tr.trainInd);
            
            %vOut = out_train(tr.valInd);
            %vTarg = ytrain(tr.valInd);
            
            %testOut = out_train(tr.testInd);
            %testTarg = ytrain(tr.testInd);
            
            %train_err_val = mean((trTarg-trOut).^2);
            %test_err_val = mean((testTarg-testOut).^2);
            %val_err_val  = mean((vTarg-vOut).^2);
            [reg_tr,~,~] = regression(trTarg, trOut);
            %[reg_val,~,~] = regression(vTarg, vOut);
            %[reg_test,~,~] = regression(testTarg, testOut)
            val_perf = mse(net,ytrain(tr.valInd),out_train(tr.valInd));
            tr_perf = mse(net,ytrain(tr.trainInd),out_train(tr.trainInd));
            bool_var = (val_perf>val_perf_reqd)||(reg_tr<reg_tr_reqd); % Training parameter
            %net
            if val_perf<val_min && tr_perf<tr_perf_min
                val_min = val_perf;
                tr_perf_min = tr_perf;
                net_min = net;
                tr_min = tr;
                %             t_train_min = ytrain(tr.trainInd);
                %             t_val_min = ytrain(tr.valInd);
                %             t_test_min = ytrain(tr.testInd);
                %             x_train_min = XTRAIN(:,tr.trainInd);
                %             x_val_min = XTRAIN(:,tr.valInd);
                %             x_test_min = XTRAIN(:,tr.testInd);
            end
            if index_watch>=max_index % this is just for the case when the iterations do not converge even after 1000 runs ....
                %             trOut = net_min(x_train_min);
                %             vOut = net_min(x_val_min);
                %             testOut = net_min(x_test_min);
                %             trTarg = t_train_min;
                %             vTarg = t_val_min;
                %             testTarg = t_test_min;
                %             net = net_min;
                %             out_train = net(XTRAIN);
                break;
            end
            index_watch = index_watch +1;
        end
        %test_k_fold = net(XTEST);
        %t_test_total_Targ = cat(2,testTarg,t_test);
        %t_test_total_Out = cat(2,testOut,test_k_fold);
        
        %train_err(random_weights) = mean((trTarg - trOut).^2);
        %test_err(random_weights) = mean((t_test_total_Targ - t_test_total_Out).^2);
        %val_err(random_weights) = mean((vTarg - vOut).^2);
        %tot_err(random_weights) = mean((ytrain-out_train).^2);
        %net_collection{random_weights} = net;
    end
    net_storage_complete{coeff_num} = net_min;
    %index_out_coeffs{coeff_num} = ind_out;
    %test_err_complete{coeff_num} = test_err;
end

save(strcat(lattice,'_final_results.mat'),'net_storage_complete')
%save('cubic_net_storage_complete24Jul.mat','net_storage_complete');
%save('cubic_test_err_complete24Jul.mat','test_err_complete');
%save('cubic_index_partial_trained24Jul.mat','index_out_coeffs');