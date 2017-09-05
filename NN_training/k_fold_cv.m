tic;
lattice = 'cubic';
load(fullfile('..','data-gen',strcat(lattice,'-data.mat'))) % xdata and ydata
load(fullfile('..','Linear',strcat('features_',lattice,'.mat'))) % xdata and ydata
X_mat = xdata;
% Loading the required files 

%load(strcat('coefficient_',lattice,'.mat')) % Its output is "coeffs variable"
%load(strcat(lattice,'-training-features-MATLAB.mat')) % Its output is "X_mat"
%load(strcat(lattice,'-non-training-feature.mat')) %Its output is "cubic_nt"
%load(strcat(lattice,'_feature_list.mat'))% Its output is feature_list
% In next few lines we convert the python indices into matlab indices 
%feature_arr = [2 9 11 13 15 17];
% 
% for i = 1:1:max(size(feature_list))
%     feature_list{i} = double(feature_list{i});
%     feature_list{i} = feature_list{i}+ones(size(feature_list{i}));
%     feature_list{i} = sort(feature_list{i});
% end
%% Here is where you need to change the things for different coefficients
coeffs = ydata;
num_coeffs = size(coeffs,2);
err_neuron_size_cell_array = cell(num_coeffs,1);
var_neuron_size_cell_array = cell(num_coeffs,1);
hidden_layer_size_store = cell(num_coeffs,1);
%cubic_nt = cubic_nt';  % Just converting the array to transpose %Now this becomes 692X13
load(fullfile('..','data-gen','cubic-non-training-data.mat'))
cubic_nt = xntdata; % 626 by 18
for coeff_num = 1:1:num_coeffs % for all the coefficients 
% 1 - C11  % 2 - C12 % 3 - C44
%% Loading the data and normalising it to [-1 ,1]
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
%% Initialising the variables 
err_neuron_size = zeros(4,11);  %The sizes of the arrays are random . MATLAB can automatically adjust
reg_neuron_size = zeros(4,11);  % These are random numbers for initialisation of arrays
var_neuron_size = zeros(4,11);
err_neuron_size2 = zeros(4,11);
reg_neuron_size2 = zeros(4,11);
var_neuron_size2 = zeros(4,11);
net_storage = cell(4,11);

%% Making a 5 fold clustering here --- randomly dividing the data set
k = 5; 
num_random_folds = k;
%% The whole process of finding the hidden layer size is repeated k times  
for random_fold = 1:1:num_random_folds
    hidden_layer_size_range = (2:15);
    N = size(t,2);  % Number of elements in target set
    indices = crossvalind('Kfold',N,k); % Making the grouping for the k fold cross validation 
    grp_set = cell(k,1);
    for ll = 1:1:k
        grp_set{ll}= find(indices==ll);
    end

    % running iterations for hidden layer size 
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
        net_cluster2 = cell(k,1);
        for cluster = 1:1:k
            trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
            % Create a Fitting Network
            %[trainInd,valInd, testInd] = divideind(81, 1:40, 41:55,56:81);
            range = (1:k);

            train_data_index = setdiff(range,(cluster));
            tr_ind_re = cat(1,grp_set{train_data_index}); %concatenating arrays
            x_train = x(:,tr_ind_re);
            t_train = t(:,tr_ind_re);

            test_data_index = (cluster);
            test_ind_re = cat(1,grp_set{test_data_index});
            x_test = x(:,test_ind_re);
            t_test = t(:,test_ind_re);

            %running the model multiple times to incorporate the effect of
            %the various inital weights conditions
            sample_test = 20;

            train_err = zeros(sample_test,1);
            test_err = zeros(sample_test,1);
            val_err = zeros(sample_test,1);
            tot_err = zeros(sample_test,1);
            net_collection = cell(sample_test,1);

            for random_weights = 1:1:sample_test
                net = fitnet(layer_size,trainFcn);
                net = configure(net,x_train, t_train);
                net.trainParam.showWindow = 0;
                net.divideParam.trainRatio = 80/100;
                net.divideParam.valRatio = 20/100;
                net.divideParam.testRatio = 0/100;
        % %         net.divideFcn = 'divideind';
        % %         size_divide = size(t_train);
        % %         num1 = floor(size_divide(2)*0.6);
        % %         num2 = floor(size_divide(2)*0.8);
        % %         net.divideParam.trainInd = (1:num1);
        % %         net.divideParam.valInd = (num1+1:num2);
        % %         net.divideParam.testInd = (num2+1:size_divide(2));

                net.performParam.regularization = 0;
                bool_var = 1;
                index_watch = 1;
                val_min = 1;
                net_min = [];
                while((bool_var))
                    init(net);
                    [net,tr] = train(net,x_train,t_train);
                    y_train = net(x_train);

                    e = gsubtract(x_train,y_train);
                    performance = perform(net,t_train,y_train);

                    if(size(t_train) ~= size(y_train))
                        t_train = t_train';
                    end
                    trTarg = t_train(tr.trainInd);
                    trOut = y_train(tr.trainInd);

                    vOut = y_train(tr.valInd);
                    vTarg = t_train(tr.valInd);

                    testOut = y_train(tr.testInd);
                    testTarg = t_train(tr.testInd);

                    train_err_val = mean((trTarg-trOut).^2);
                    test_err_val = mean((testTarg-testOut).^2);
                    val_err_val  = mean((vTarg-vOut).^2);
                    [reg_tr,~,~] = regression(trTarg, trOut);
                    [reg_val,~,~] = regression(vTarg, vOut);
                    [reg_test,~,~] = regression(testTarg, testOut);
                    
                    bool_var = (val_err_val>0.05)||(reg_tr<0.85); % Training parameter
                    
                    if val_err_val<val_min
                        val_min = val_err_val;
                        net_min = net;
                        t_train_min = t_train(tr.trainInd);
                        t_val_min = t_train(tr.valInd);
                        t_test_min = t_train(tr.testInd);
                        x_train_min = x_train(:,tr.trainInd);
                        x_val_min = x_train(:,tr.valInd);
                        x_test_min = x_train(:,tr.testInd);
                    end
                    if index_watch>1000 % this is just for the case when the iteration do not converger
                        %even after 1000 runs ....
                        trOut = net_min(x_train_min);
                        vOut = net_min(x_val_min);
                        testOut = net_min(x_test_min);
                        trTarg = t_train_min;
                        vTarg = t_val_min;
                        testTarg = t_test_min;
                        net = net_min;
                        y_train = net(x_train);
                        break;
                    end 
                    index_watch = index_watch +1;
                end
                test_k_fold = net(x_test);
                t_test_total_Targ = cat(2,testTarg,t_test);
                t_test_total_Out = cat(2,testOut,test_k_fold);

                train_err(random_weights) = mean((trTarg - trOut).^2);
                test_err(random_weights) = mean((t_test_total_Targ - t_test_total_Out).^2);
                val_err(random_weights) = mean((vTarg - vOut).^2);
                tot_err(random_weights) = mean((t_train-y_train).^2);
                net_collection{random_weights} = net;
            end
            [valu,indx] = min(test_err);

            net_final1 = net_collection{indx};
            test_err_cluster(cluster) = valu;
            net_cluster{cluster} = net_final1;
            test_var_cluster(cluster) = var((t_test-net_final1(x_test)).^2);
            test_reg_cluster(cluster) = regression(t_test,net_final1(x_test));
            cluster
        end
        err_neuron_size(random_fold,layer_size) = mean(test_err_cluster); 
        var_neuron_size(random_fold,layer_size) = mean(test_var_cluster);
        reg_neuron_size(random_fold,layer_size) = mean(test_reg_cluster);
        [finalval, finalindx] = min(test_err_cluster);
        net_storage{random_fold,layer_size} = net_cluster{finalindx};
    end
end
%% Post processing

mean_err = mean(err_neuron_size((1:random_fold), 1:15),1);

[val_err,ind_err] = min(mean_err(hidden_layer_size_range));
ind_exact = ind_err + (hidden_layer_size_range(1)-1);

hidden_layer_size_store{coeff_num} = ind_exact;  
%finding the max regression of all network with minimum cluster error
err_neuron_size_cell_array{coeff_num} = err_neuron_size;
var_neuron_size_cell_array{coeff_num} = var_neuron_size;
%finally save this Coeff_predicted variable as your respective variable


end
save(strcat('hidden_layersize_',lattice,'.mat'),'hidden_layer_size_store');
save(strcat(lattice,'_err_neuron_size.mat'),'err_neuron_size_cell_array');
save(strcat(lattice,'_var_neuron_size.mat'),'var_neuron_size_cell_array');
toc;