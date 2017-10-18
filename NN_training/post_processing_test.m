clear all

%% Here you should put the predicted valued file with non_training data.
lattice = strtrim(fileread('lattice-type.txt'));
load(fullfile('..','data-gen',strcat(lattice,'-non-training-data.mat')));
mps_nt = mps;
load(strcat(lattice,'_final_results.mat'));
load(fullfile('..','data-gen',strcat(lattice,'-data.mat')));
load(fullfile('..','Linear',strcat('features_',lattice,'.mat'))) ;
load(fullfile('..','Linear',strcat('features_',lattice,'.mat'))) ;

% In next few lines we convert the python indices into matlab indices
num_samples = 1000;
X_mat = xdata(1:end-floor(0.1*size(xdata,1)),:);
coeffs = ydata(1:end-floor(0.1*size(xdata,1)),:);
cubic_nt = xntdata;

% test data
xtest = xdata(end-floor(0.1*size(xdata,1))+1:end,:);
coeffstest = ydata(end-floor(0.1*size(xdata,1))+1:end,:);
%cubic_nt = xdata;

%%
sz_nt = size(cubic_nt,1);
sz_tt = size(xtest,1);
num_coeffs = size(coeffs,2);
perf = zeros(num_samples, num_coeffs);
Coeffs_cell = cell(num_samples, num_coeffs);
predcoeffs = zeros(sz_nt, num_coeffs, num_samples);
predcoeffs_test = zeros(sz_tt, num_coeffs, num_samples);
test_perf = zeros(num_coeffs, num_samples);
for coeff_num = 1:num_coeffs
    feature_arr = feature_list{coeff_num};
    X1 = X_mat(:,feature_arr);
    non_training_yy = cubic_nt(:,feature_arr)';
    coeff_num
    xtest_tp = xtest(:,feature_arr)';
    yy = X1'; %Doing transpose to feed into Neural network
    %Mapminmax starts below to map x from [-1,1]
    [x, tot_inp_recover] = mapminmax(yy);
    % tot_inp is our favorable output and tot_inp_recover is the structure by
    % which we can apply that transformation to other values as well and can
    % also get the original value of tot_inp.
    non_training = mapminmax('apply',non_training_yy,tot_inp_recover);
    xtest_mapped = mapminmax('apply',xtest_tp,tot_inp_recover);
    net_storage = net_storage_complete{coeff_num};
    tr_storage = tr_storage_complete{coeff_num};
    t_yy = coeffs(:,coeff_num)'; % t here refers to the target in ANNs
    [t, t_recover] = mapminmax(t_yy); %Mapping the coefficients as well in [-1,1]
    
    for num_samp = 1:num_samples
        tr = tr_storage{num_samp};
        net = net_storage{num_samp};
        %tr.perf
        %tr.best_perf
        %tr.perf(end)
        perf(num_samp,coeff_num) = tr.perf(end);
        Coeffs_mat = net(non_training);
        Coeffs_test_mat = net(xtest_mapped);
        Coeffs_mat = mapminmax('reverse',Coeffs_mat,t_recover);
        Coeffs_test_mat = mapminmax('reverse',Coeffs_test_mat,t_recover);
        predcoeffs(:,coeff_num, num_samp) = Coeffs_mat;
        predcoeffs_test(:, coeff_num, num_samp) = Coeffs_test_mat;
        
        test_perf(coeff_num, num_samp) = 1 - sum((coeffstest(:,coeff_num) - predcoeffs_test(:,coeff_num,num_samp)).^2)/sum((coeffstest(:,coeff_num) - mean(coeffstest(:,coeff_num))).^2);
    end
end

%% generating model choices and good nets
ngoodnets = zeros(1,num_coeffs);
sortedtestperf = cell(1,num_coeffs);
sortind = cell(1, num_coeffs);
for i=1:num_coeffs
    ngoodnets(i) = length(find(index_out_coeffs{i}==0));
    ngoodnets(i) = length(find(test_perf(i,:)>0.0));
    [sortedtestperf{i},sortind{i}] = sort(test_perf(i,:),'descend');
end

ngoodnets

%minbest = min(ngoodnets);

%sortNchoose = zeros(minbest,num_coeffs);
%Coeffs_mod = zeros(num_coeffs,minbest,sz_nt);

% for i=1:num_coeffs
%     ind_full_val = find(index_out_coeffs{i}==0);
%     perf_full = perf(ind_full_val,i);
%     [val, sortind] = sort(perf_full);
%     %sortNchoose(:,i) = ind_full_val(sortind(1:minbest));
% end

combs = 10000;
%cubic_mat = zeros(6,6,sz_nt,combs); -- not storing right now
G_v = zeros(sz_nt,combs);
G_r = zeros(sz_nt,combs);
B_v = zeros(sz_nt,combs);
B_r = zeros(sz_nt,combs);
Nu = zeros(sz_nt,combs);
G = zeros(sz_nt,combs);
B = zeros(sz_nt,combs);

nue = 0.42;
VM = 1.3e-05;
Ge = 3.4e09;
gamma = 0.556;

chi_new = zeros(sz_nt,combs);
is_posdef = zeros(sz_nt, combs);
chipres = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (VM-VMc).*(Gs.^2.*(3-4.*nue)+Ge.^2.*(-3+4.*nus)).*k./(2.*z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
chisurf = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) -(VM+VMc).*gamma.*k.^2./2./z;
chitau = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (2.*Ge.*Gs.*(VM + VMc).*(2 -3.*nus + nue.*(-3 + 4.*nus)).*k)./(z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));

model_choice = zeros(num_coeffs, combs);
for i=1:num_coeffs
    %ll = 1;
    %ul = ngoodnets(i);
    %model_choice(i,:) = randi([1,ngoodnets(i)],1,combs);
    model_choice(i,:) = datasample( sortind{i}(1:ngoodnets(i)) , combs);
end

%%
Gnew = cell(1,sz_nt);
Nunew = cell(1,sz_nt);
ind_posdefn = cell(1,sz_nt);
Cmatrix = cell(sz_nt,combs);
for mat=1:sz_nt
    mat
    for i=1:combs
        coeffs_model = [];
        for j=1:num_coeffs
            coeffs_model = [coeffs_model, predcoeffs(mat, j, model_choice(j,i))];
        end
        Cmat = constructC(lattice, coeffs_model);
        Cmatrix{mat,i} = Cmat;
        [~,p] = chol(Cmat);
        if (det(Cmat)~=0)
            [~, p] = chol(Cmat);
            if(p==0)
                is_posdef(mat,i) = 1;
            end
            Smat = inv(Cmat);
            G_v = (Cmat(1,1)+Cmat(2,2)+Cmat(3,3)+3*(Cmat(4,4)+Cmat(5,5)+Cmat(6,6))-Cmat(1,2)-Cmat(2,3)-Cmat(3,1))/15;
            B_v = (Cmat(1,1)+Cmat(2,2)+Cmat(3,3)+2*(Cmat(1,2)+Cmat(2,3)+Cmat(3,1)))/9;
            B_r = 1/(Smat(1,1)+Smat(2,2)+Smat(3,3)+2*(Smat(1,2)+Smat(2,3)+Smat(3,1)));
            G_r = 15/(4*(Smat(1,1)+Smat(2,2)+Smat(3,3))+3*(Smat(4,4)+Smat(5,5)+Smat(6,6))-4*(Smat(1,2)+Smat(2,3)+Smat(3,1)));
        else
            G_v = 0;
            G_r = 0;
            B_v = 0;
            B_r = 0;
        end
        
        B(mat,i) = 0.5*(B_v+B_r)*10^9;
        G(mat,i) = 0.5*(G_v+G_r)*10^9;
        ind_new = (1:sz_nt);
        Nu(mat, i) = (3.*B(mat, i) - 2.*G(mat, i))./(6.*B(mat, i) + 2.*G(mat, i));
        vol_new = volrat(ind_new);
        chi = chipres(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),10^8)+chitau(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),10^8, gamma);
        chi_new(mat,i) = chi./10^12;
    end
    ind_posdefn{mat} = find(is_posdef(mat,:)==1);
    Gnew{mat} = G(mat,ind_posdefn{mat});
    Nunew{mat} = Nu(mat,ind_posdefn{mat});
end

%% Creating normal distribution
prob_stable = zeros(1,sz_nt);
meanG = zeros(1,sz_nt);
stdG = zeros(1,sz_nt);
meanNu = zeros(1,sz_nt);
stdNu = zeros(1,sz_nt);
chimean = zeros(1,sz_nt);
for mat = 1:sz_nt
    neg = size(find(chi_new(mat,ind_posdefn{mat})<0),2);
    tot = size(chi_new(mat,ind_posdefn{mat}),2);
    prob_stable(mat) = neg/tot;
    meanG(mat) = mean(Gnew{mat});
    stdG(mat) = std(Gnew{mat});
    meanNu(mat) = mean(Nunew{mat});
    stdNu(mat) = std(Nunew{mat});
    chimean(mat) = mean(chi_new(mat,ind_posdefn{mat}));
end
save(strcat(lattice,'_post_results.mat'), 'chimean', 'meanG', 'prob_stable', 'stdG', 'ngoodnets','ind_posdefn','Cmatrix');

% %%
histogram(meanG/10^9,'NumBins',25)
ax1=gca;
set(ax1,'Box','on')
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'\textbf{Shear modulus/GPa}','Interpreter','latex','FontWeight','bold','FontSize',24,'Fontname','Times New Roman');
ylabel(ax1,'# of materials','Interpreter','latex','FontWeight','bold','FontSize',24,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
%xlim([120,200])
% 

