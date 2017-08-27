tic;
load coefficient_cubic.mat
load cubic-training-features-MATLAB.mat
load cubic-non-training-feature-normalized.mat
load cubic_feature_multiplier.mat
load cubic_feature_indices.mat

%% Here is where you need to change the things for different coefficients
coeff_num = 1;
% 1 - C11
% 2 - C12
% 3 - C44
%%
feature_mat = feature_ind{coeff_num};
feature_mult = coeff_multiplier(coeff_num);
X1 = X_mat(:,feature_mat);
non_training = cubic_nt(:,feature_mat)';
x = X1';
tot_inp = [x,non_training];
[tot_inp, tot_inp_recover] = mapminmax(tot_inp);
siz_x = size(x);
x = tot_inp(:,1:siz_x(2));
non_training = tot_inp(:,siz_x(2)+1:end);
t = coeffs(:,coeff_num)';

[t, t_recover] = mapminmax(t);
Y = t';
err_neuron_size = zeros(4,11);
reg_neuron_size = zeros(4,11);
var_neuron_size = zeros(4,11);
net_storage = cell(4,11);
%% Making a 9 fold clustering here --- randomly dividing the data set
k = 5; 
%% Dividing into the sets randomly 
for random_fold = 1:1:4
hidden_layer_size_range = (4:15);
sz = size(t);
N = sz(2);  %Number of elements in target set
indixes = crossvalind('Kfold',N,k);
grp_set = cell(k,1);
for ll = 1:1:k
    grp_set{ll}= find(indixes==ll);
end

index_layer = 1;
for layer_size = hidden_layer_size_range
    test_err_cluster = zeros(k,1);
    test_reg_cluster = zeros(k,1);
    test_var_cluster = zeros(k,1);
    test_err_cluster2 = zeros(k,1);
    test_reg_cluster2 = zeros(k,1);
    test_var_cluster2 = zeros(k,1);
    layer_size
    random_fold
    net_cluster = cell(k,1);
for cluster = 1:1:k
    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
    % Create a Fitting Network
    %[trainInd,valInd, testInd] = divideind(81, 1:40, 41:55,56:81);
    
    range = (1:k);
    train_data_index = setdiff(range,(cluster));
    tr_ind_re = cat(1,grp_set{train_data_index});
    x_train = x(:,tr_ind_re);
    t_train = t(:,tr_ind_re);
    test_data_index = (cluster);
    test_ind_re = cat(1,grp_set{test_data_index});
    x_test = x(:,test_ind_re);
    t_test = t(:,test_ind_re);
    %running the model for a distribution of k fold thing for multiple times to incorporate the effect of the various inital weights condition d
    sample_test = 5;
    train_err = zeros(sample_test,1);
    val_err = zeros(sample_test,1);
    net_collection = cell(sample_test,1);
    reg_tr_before = 0.94;
    reg_val_before = 0.9;
    for random_weights = 1:1:sample_test
        random_weights
        net = fitnet(layer_size,trainFcn);
        net = configure(net,x_train, t_train);
        net.trainParam.showWindow = 0;
        net.divideParam.trainRatio = 60/100;
        net.divideParam.valRatio = 20/100;
        net.divideParam.testRatio = 20/100;
% %         net.divideFcn = 'divideind';
% %         size_divide = size(t_train);
% %         num1 = floor(size_divide(2)*0.6);
% %         num2 = floor(size_divide(2)*0.8);
% %         net.divideParam.trainInd = (1:num1);
% %         net.divideParam.valInd = (num1+1:num2);
% %         net.divideParam.testInd = (num2+1:size_divide(2));
        net.performParam.regularization = 0;
        bool_var = 1;
        while(bool_var)
            init(net);
            [net,tr] = train(net,x_train,t_train);
            y_train = net(x_train);

            e = gsubtract(t_train,y_train);
            performance = perform(net,t_train,y_train)

            if(size(t_train) ~= size(y_train))
                t_train = t_train';
            end
            trTarg = t_train(tr.trainInd);
            trOut = y_train(tr.trainInd);

            vOut = y_train(tr.valInd);
            vTarg = t_train(tr.valInd);
            
            testOut = y_train(tr.testInd);
            testTarg = t_train(tr.testInd);
            
            [reg_tr,~,~] = regression(trTarg, trOut);
            [reg_val,~,~] = regression(vTarg, vOut);
            [reg_test,~,~] = regression(testTarg, testOut);
            
            bool_var = (performance>0.02)||(reg_val<0.93)||(reg_tr<0.93);
        end
        reg_val_before = reg_val;
        train_err(random_weights) = mean((trTarg - trOut).^2);
        test_err(random_weights) = mean((testTarg - testOut).^2);
        val_err(random_weights) = mean((vTarg - vOut).^2);
        tot_err(random_weights) = mean((t_train-y_train).^2);
        net_collection{random_weights} = net;
    end
    [valu,indx] = min(test_err);
    [valvalu, valindx] = min(val_err);
    net_final1 = net_collection{indx};
    test_err_cluster(cluster) = mean((t_test-net_final1(x_test)).^2);
    test_var_cluster(cluster) = var((t_test-net_final1(x_test)).^2);
    test_reg_cluster(cluster) = regression(t_test,net_final1(x_test));
    net_final2 = net_collection{valindx};
    test_err_cluster2(cluster) = mean((t_test-net_final2(x_test)).^2);
    test_var_cluster2(cluster) = var((t_test-net_final2(x_test)).^2);
    test_reg_cluster2(cluster) = regression(t_test,net_final2(x_test));
    cluster
    net_cluster{cluster} = net_final1;
end
    err_neuron_size(random_fold,layer_size) = mean(test_err_cluster); 
    var_neuron_size(random_fold,layer_size) = mean(test_var_cluster);
    reg_neuron_size(random_fold,layer_size) = mean(test_reg_cluster);
    [finalval, finalindx] = min(0.75.*test_err_cluster+0.25.*test_var_cluster);
    net_storage{random_fold,layer_size} = net_cluster{finalindx};
    index_layer = index_layer+1;
end
end
err_neuron_size
mean_err = mean(err_neuron_size([1,2,3,4], hidden_layer_size_range));
[val_err,ind_err] = min(mean_err);
%finding the max regression of all network with minimum cluster error

[valout, valindex] = max([regression(t,net_storage{1,ind_err}(x)) ,regression(t,net_storage{2,ind_err}(x))...
    ,regression(t,net_storage{3,ind_err}(x)),regression(t,net_storage{4,ind_err}(x))]);
net_final_out = net_storage{valindex, ind_err};

Coeff_predicted = net_final_out(non_training);
Coeff_predicted = mapminmax('reverse',Coeff_predicted,t_recover);
Coeff_predicted = Coeff_predicted.*feature_mult;

if coeff_num == 1
    C11_non_training = Coeff_predicted;
elseif coeff_num == 2
    C12_non_training = Coeff_predicted;
elseif coeff_num == 3
    C44_non_training = Coeff_predicted; 
end
%finally save this Coeff_predicted variable as your respective variable
toc;