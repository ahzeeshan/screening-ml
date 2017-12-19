function [ yfit ] = neural_net_ps(XTRAIN,ytrain,XTEST,layer_size, sample_test, max_index, Rsq_val_reqd, Rsq_tr_reqd)
% This function returns yfit of the XTEST corresponding to neural network
% model
trainFcn = 'trainlm';
XTRAIN = XTRAIN';
ytrain = ytrain';
XTEST = XTEST';
%val_min = 1;
%tr_perf_min = 1;
%net_min = [];
net_min = cell(1,sample_test);
val_min = ones(sample_test);
parfor random_weights = 1:1:sample_test
    net = fitnet(layer_size,trainFcn);
    net = configure(net,XTRAIN,ytrain);
    net.trainParam.showWindow = 0;
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 0/100;
    net.performParam.regularization = 0;
    bool_var = 1;
    index_watch = 1;
    %    val_min(random_weights) = 1;
    %net_min{random_weights} = [];
    while((bool_var))
        init(net);
        [net,tr] = train(net,XTRAIN,ytrain);
        out_train = net(XTRAIN);
        %e = gsubtract(XTRAIN,ytrain);
        %performance = perform(net,ytrain,out_train);
        %   if(size(ytrain) ~= size(out_train))
        %    ytrain = ytrain';
        %end
        %trTarg = ytrain(tr.trainInd);
        %trOut = out_train(tr.trainInd);
        %[reg_tr,~,~] = regression(trTarg, trOut);
        Rsq_tr = 1 - sum((ytrain(tr.trainInd)- ...
                          out_train(tr.trainInd)).^2)/sum((ytrain(tr.trainInd) - mean(ytrain(tr.trainInd)) ).^2  );
        Rsq_val = 1 - sum( (ytrain(tr.valInd) - out_train(tr.valInd)).^2 ...
                           )/sum( (ytrain(tr.valInd) - ...
                                   mean(ytrain(tr.valInd)) ).^2  );
        val_perf = mse(net,ytrain(tr.valInd),out_train(tr.valInd));
        tr_perf = mse(net,ytrain(tr.trainInd),out_train(tr.trainInd));
        %        bool_var = (val_perf>val_perf_reqd)||(reg_tr<reg_tr_reqd); % Training parameter
        bool_var = (Rsq_val < Rsq_val_reqd) || (Rsq_tr < Rsq_tr_reqd);
        if val_perf<val_min(random_weights) || index_watch==1 %&& tr_perf<tr_perf_min
            val_min(random_weights) = val_perf; 
            %tr_perf_min = tr_perf;
            net_min{random_weights} = net;
            %tr_min = tr;
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
[~,sortrw] = sort(val_min);
yfit = net_min{sortrw(1)}(XTEST);
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
