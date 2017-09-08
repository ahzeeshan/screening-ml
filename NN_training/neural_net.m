function [ yfit ] = neural_net(XTRAIN,ytrain,XTEST,layer_size, sample_test, max_index, val_perf_reqd, reg_tr_reqd)
% This function returns yfit of the XTEST corresponding to neural network
% model
trainFcn = 'trainlm';
XTRAIN = XTRAIN';
ytrain = ytrain';
XTEST = XTEST';
val_min = 1;
tr_perf_min = 1;
net_min = [];
for random_weights = 1:1:sample_test
    net = fitnet(layer_size,trainFcn);
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
%net_min
%mse(net,ytrain(tr_min.trainInd),out_train(tr_min.trainInd))
yfit = net_min(XTEST);
yfit = yfit';
end

% Diagnostics
% tr.vperf
% tr.tperf
% plotperf(tr)
% hold on
% plotregression(trTarg, trOut, 'Train', vTarg, vOut, 'Validation')
% val_perf
% reg_tr