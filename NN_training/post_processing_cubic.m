clear all
%close all
%clc
%% Here you should put the predicted valued file with non_training data.
lattice = fileread('lattice-type.txt');
load(fullfile('..','data-gen',strcat(lattice,'-non-training-data.mat')));
load(strcat(lattice,'_final_results.mat'));
load(fullfile('..','data-gen',strcat(lattice,'-data.mat')));
load(fullfile('..','Linear',strcat('features_',lattice,'.mat'))) ;
load(fullfile('..','Linear',strcat('features_',lattice,'.mat'))) ;

% In next few lines we convert the python indices into matlab indices
num_samples = 1000;
X_mat = xdata;
coeffs = ydata;
cubic_nt = xntdata;
%cubic_nt = cubic_nt';

%%
num_coeffs = size(coeffs,2);
%non_training = cell(num_coeffs,1);
%tot_inp_recover = cell(num_coeffs,1);
%t_recover = cell(num_coeffs,1);
perf = zeros(num_samples, num_coeffs);
Coeffs_cell = cell(num_samples, num_coeffs);
for coeff_num = 1:1:num_coeffs
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
    net_storage = net_storage_complete{coeff_num};
    tr_storage = tr_storage_complete{coeff_num};
    t_yy = coeffs(:,coeff_num)'; % t here refers to the target in ANNs
    [t, t_recover] = mapminmax(t_yy); %Mapping the coefficients as well in [-1,1]
    
    for num_samp = 1:num_samples
        tr = tr_storage{num_samp};
        net = net_storage{num_samp};
        perf(num_samp,coeff_num) = tr.best_perf;
        Coeffs_mat = net(non_training);
        Coeffs_mat = mapminmax('reverse',Coeffs_mat,t_recover);
        Coeffs_cell{num_samp,coeff_num} = Coeffs_mat;
    end
end
%%
N_choose = 18;
k_choose = 9;
size_consider = N_choose; % Lowest N_choose test errrors were taken

sz_nt = size(cubic_nt,1);
test_err = zeros(num_samples,num_coeffs);
sortNchoose = zeros(size_consider,num_coeffs);
Coeffs_mod = zeros(num_coeffs,size_consider,sz_nt);
%% Finding the ones with positive definite matrix

% for i = 1:1:num_samples
%     C11_matrix = Coeffs_cell{i,1};
%     C12_matrix = Coeffs_cell{i,2};
%     C44_matrix = Coeffs_cell{i,3};
%     is_posdef(i) = check_g_vals_all_element_cubic(C11_matrix, C12_matrix, C44_matrix);
% end
%%
% for i=1:num_coeffs
% perf(find(ind_full_val==0),:)
% end
for i=1:num_coeffs
    ind_full = index_out_coeffs{i};
    ind_full_val = find(ind_full==0);
    perf_full = perf(ind_full_val,i);
    [val, sorte] = sort(perf_full);
    sortNchoose(:,i) = ind_full_val(sorte(1:size_consider));
    for j = 1:1:size_consider
        Coeffs_mod(i,j,:) = Coeffs_cell{sortNchoose(j,i),i};
    end
end



% combs = nchoosek(N_choose, k_choose);
% CombEle = combnk((1:N_choose), k_choose);
% Coeffs_ensem = zeros(num_coeffs,combs, sz_nt); % here 5 is random number so that we can initialize the array
% 
% for num = 1:1:combs
%     for k = 1:1:num_coeffs
%         Coeffs_ensem(k,num,:) = mean(Coeffs_mod(k,CombEle(num,:),:),2);
%     end
% end

Coeffs_ensem = Coeffs_mod;
combs = size_consider;
sz_put = combs;
cubic_mat = zeros(6,6,sz_nt);
G_v = zeros(sz_nt,1);
G_r = zeros(sz_nt,1);
B_v = zeros(sz_nt,1);
B_r = zeros(sz_nt,1);
nu = zeros(sz_nt,sz_put);
G = zeros(sz_nt, sz_put);
B = zeros(sz_nt, sz_put);

nue = 0.42;
VM = 1.3e-05;
Ge = 3.4e09;

chi_new = zeros(sz_nt,sz_put);
is_posdef = zeros(num_samples,1);

for ns = 1:1:sz_put
    for i = 1:1:sz_nt
        Cmat = constructC(lattice, Coeffs_ensem(:,ns,i));
        [~,p] = chol(Cmat);
        cubic_mat(:,:,i) = Cmat; % purely for storage
        if (det(cubic_mat(:,:,i))~=0)
            [~, p] = chol(cubic_mat(:,:,i));
            if(p==0)
                is_posdef(i) = 1;
            end
            Smat = inv(Cmat);
            G_v(i) = (Cmat(1,1)+Cmat(2,2)+Cmat(3,3)+3*(Cmat(4,4)+Cmat(5,5)+Cmat(6,6))-Cmat(1,2)-Cmat(2,3)-Cmat(3,1))/15;
            B_v(i) = (Cmat(1,1)+Cmat(2,2)+Cmat(3,3)+2*(Cmat(1,2)+Cmat(2,3)+Cmat(3,1)))/9;
            B_r(i) = 1/(Smat(1,1)+Smat(2,2)+Smat(3,3)+2*(Smat(1,2)+Smat(2,3)+Smat(3,1)));
            G_r(i) = 15/(4*(Smat(1,1)+Smat(2,2)+Smat(3,3))+3*(Smat(4,4)+Smat(5,5)+Smat(6,6))-4*(Smat(1,2)+Smat(2,3)+Smat(3,1)));
        else
            G_v(i) = 0;
            G_r(i) = 0;
            B_v(i) = 0;
            B_r(i) = 0;
        end
    end
    B(:,ns) = 0.5*(B_v+B_r)*10^9;
    G(:,ns) = 0.5*(G_v+G_r)*10^9;
    indG = find(G(:,ns)<=0);
    indB = find(B(:,ns)<=0);
    gamma = 0.556;
    %ind_new = union(indG,indB);
    %ind_new = setdiff((1:692),ind_new);
    ind_new = (1:sz_nt);
    nu(:,ns) = (3.*B(:,ns) - 2.*G(:,ns))./(6.*B(:,ns) + 2.*G(:,ns));
    
    vol_new = volrat(ind_new);
    chipres = @(VM,VMc,z,Gs,Ge,nue,nus,k) (VM-VMc).*(Gs.^2.*(3-4.*nue)+Ge.^2.*(-3+4.*nus)).*k./(2.*z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
    chitau = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (2.*Ge.*Gs.*(VM + VMc).*(2 -3.*nus + nue.*(-3 + 4.*nus)).*k)./(z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
    
    for i = 1:1:sz_nt
        %chi(i) = chipres(VM,vol_new(i)*VM,1,G(ind_new(i)),Ge,nue,mu(i),10^8)+...
        %    chitau(VM,vol_new(i)*VM,1,G(ind_new(i)),Ge,nue,mu(i),10^8, gamma);
        chi(i) = chipres(VM,vol_new(i)*VM,1,G(i,ns),Ge,nue,nu(i,ns),10^8)+...
            chitau(VM,vol_new(i)*VM,1,G(i,ns),Ge,nue,nu(i,ns),10^8, gamma);
    end
    chi_new(:,ns) = chi./10^12;
end
%% Creating normal distribution
prob_stable = zeros(1,sz_nt);
mean_vals = zeros(1,sz_nt);
std_vals = zeros(1,sz_nt);
for i = 1:1:sz_nt
    [abc, def] = sort(abs(chi_new(i,:)),'descend');
    numb = floor(0.1*combs);
    chill = chi_new(i,def(numb:end));
    neg = max(size(find(chill<0)));
    tot = max(size(chill));
    prob_stable(i) = neg/tot; % Computes cdf less at zero which is the boundary of stability
end