clear all
close all
clc
%% Here you should put the predicted valued file with non_training data.
lattice = 'cubic'
load(fullfile('..','data-gen',strcat(lattice,'-non-training-data.mat')));
load(strcat(lattice,'_final_results.mat'));
load(fullfile('..','data-gen',strcat(lattice,'-data.mat')));
load(fullfile('..','Linear',strcat('features_',lattice,'.mat'))) ;

% In next few lines we convert the python indices into matlab indices
num_samples = 1;
X_mat = xdata;
coeffs = ydata;
cubic_nt = xntdata;
cubic_nt = cubic_nt';

%%
num_coeffs = size(coeffs,2);
non_training = cell(num_coeffs,1);
tot_inp_recover = cell(num_coeffs,1);
t_recover = cell(num_coeffs,1);

Coeffs_cell = cell(num_samples, num_coeffs);
for coeff_num = 1:1:num_coeffs
    feature_arr = feature_list{coeff_num};
    
    X1 = X_mat(:,feature_arr);
    non_training_yy{coeff_num} = cubic_nt(:,feature_arr)';
    
    yy = X1'; %Doing transpose to feed into Neural network
    
    %Mapminmax starts below to map x from [-1,1]
    
    [x, tot_inp_recover{coeff_num}] = mapminmax(yy);
    % tot_inp is our favorable output and tot_inp_recover is the structure by
    % which we can apply that transformation to other values as well and can
    % also get the original value of tot_inp.
    non_training{coeff_num} = mapminmax('apply',non_training_yy{coeff_num},tot_inp_recover{coeff_num});
    net_storage = net_storage_complete{coeff_num};
    t_yy = coeffs(:,coeff_num)'; % t here refers to the target in ANNs
    [t, t_recover{coeff_num}] = mapminmax(t_yy); %Mapping the coefficients as well in [-1,1]
    
    for num_samp = 1:num_samples
        net = net_storage{num_samp};
        Coeffs_mat = net(non_training{coeff_num});
        Coeffs_mat = mapminmax('reverse',Coeffs_mat,t_recover{coeff_num});
        Coeffs_cell{num_samp,coeff_num} = Coeffs_mat;
    end
end
%%
N_choose = 18;
k_choose = 9;
size_consider = N_choose; % Lowest N_choose test errrors were taken

sz_nt = size(cubic_nt);
test_err = zeros(num_samples,num_coeffs);
sortNchoose = zeros(size_consider,num_coeffs);
Coeffs_mod = zeros(num_coeffs,size_consider,max(sz_nt));
%% Finding the ones with positive definite matrix
is_posdef = zeros(num_samples,1);
for i = 1:1:num_samples
    C11_matrix = Coeffs_cell{i,1};
    C12_matrix = Coeffs_cell{i,2};
    C44_matrix = Coeffs_cell{i,3};
    is_posdef(i) = check_g_vals_all_element_cubic(C11_matrix, C12_matrix, C44_matrix);
end
%%
for i=1:1:num_coeffs
    test_err(:,i) = test_err_complete{i};
    mat = test_err_complete{i};
    ind_full = index_out_coeffs{i};
    ind_full_val = find(ind_full==0); % which were trained in 1000 sims
    
    test_err_full = mat(ind_full_val);
    %[val, sorte] = sort(test_err(:,i));
    [val, sorte] = sort(test_err_full);
    sortNchoose(:,i) = ind_full_val(sorte(1:size_consider));
    for j = 1:1:size_consider
        Coeffs_mod(i,j,:) = Coeffs_cell{sortNchoose(j,i),i};
    end
end
combs = nchoosek(N_choose, k_choose);
CombEle = combnk((1:N_choose), k_choose);
Coeffs_ensem = zeros(num_coeffs,combs, max(sz_nt)); % here 5 is random number so that we can initialize the array

for num = 1:1:combs
    for k = 1:1:num_coeffs
        Coeffs_ensem(k,num,:) = mean(Coeffs_mod(k,CombEle(num,:),:),2);
    end
end

sz_put = combs;
cubic_mat = zeros(6,6,max(sz_nt));
G_v = zeros(max(sz_nt),1);
G_r = zeros(max(sz_nt),1);
B_v = zeros(max(sz_nt),1);
B_r = zeros(max(sz_nt),1);
mu = zeros(max(sz_nt),sz_put);
G = zeros(max(sz_nt), sz_put);
B = zeros(max(sz_nt), sz_put);

chi_new = zeros(max(sz_nt),sz_put);
for ns = 1:1:sz_put
    C11_nt = Coeffs_ensem(1,ns,:);
    C12_nt = Coeffs_ensem(2,ns,:);
    C44_nt = Coeffs_ensem(3,ns,:);
    
    for i = 1:1:max(sz_nt)
        C11_mat = C11_nt(i);
        C12_mat = C12_nt(i);
        C44_mat = C44_nt(i);
        cubic_mat(:,:,i) = [C11_mat ,C12_mat,C12_mat,0      ,0      ,0;...
            C12_mat ,C11_mat,C12_mat,0      ,0      ,0;...
            C12_mat ,C12_mat,C11_mat,0      ,0      ,0;...
            0       ,0      ,0      ,C44_mat,0      ,0;...
            0       ,0      ,0      ,0      ,C44_mat,0;...
            0       ,0      ,0      ,0      ,0      ,C44_mat];
        C11 = cubic_mat(1,1,i);
        C22 = cubic_mat(2,2,i);
        C33 = cubic_mat(3,3,i);
        C44 = cubic_mat(4,4,i);
        C55 = cubic_mat(5,5,i);
        C66 = cubic_mat(6,6,i);
        C12 = cubic_mat(1,2,i);
        C23 = cubic_mat(2,3,i);
        C31 = cubic_mat(3,1,i);
        if (det(cubic_mat(:,:,i))~=0)
            s_mat = inv(cubic_mat(:,:,i));
            s11 = s_mat(1,1);
            s22 = s_mat(2,2);
            s33 = s_mat(3,3);
            s44 = s_mat(4,4);
            s55 = s_mat(5,5);
            s66 = s_mat(6,6);
            s12 = s_mat(1,2);
            s23 = s_mat(2,3);
            s31 = s_mat(3,1);
            G_v(i) = (C11+C22+C33+3*(C44+C55+C66)-C12-C23-C31)/15;
            B_v(i) = (C11+C22+C33+2*(C12+C23+C31))/9;
            B_r(i) = (1)/(s11+s22+s33+2*(s12+s23+s31));
            G_r(i) = 15/(4*(s11+s22+s33)+3*(s44+s55+s66)-4*(s12+s23+s31));
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
    ind_new = (1:max(sz_nt));
    mu(:,ns) = (3.*B(:,ns) - 2.*G(:,ns))./(6.*B(:,ns) + 2.*G(:,ns));
    nue = 0.42;
    vol_new = vol_ratio(ind_new);
    chipres = @(VM,VMc,z,Gs,Ge,nue,nus,k) (VM-VMc).*(Gs.^2.*(3-4.*nue)+Ge.^2.*(-3+4.*nus)).*k./(2.*z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
    chitau = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (2.*Ge.*Gs.*(VM + VMc).*(2 -3.*nus + nue.*(-3 + 4.*nus)).*k)./(z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
    
    %calculating boundary
    VM = 1.3e-05;
    Ge = 3.4e09;
    for i = 1:1:max(sz_nt)
        %chi(i) = chipres(VM,vol_new(i)*VM,1,G(ind_new(i)),Ge,nue,mu(i),10^8)+...
        %    chitau(VM,vol_new(i)*VM,1,G(ind_new(i)),Ge,nue,mu(i),10^8, gamma);
        chi(i) = chipres(VM,vol_new(i)*VM,1,G(i,ns),Ge,nue,mu(i,ns),10^8)+...
            chitau(VM,vol_new(i)*VM,1,G(i,ns),Ge,nue,mu(i,ns),10^8, gamma);
    end
    chi_new(:,ns) = chi./10^12;
end
%% Creating normal distribution
prob_stable = zeros(1,max(sz_nt));
mean_vals = zeros(1,max(sz_nt));
std_vals = zeros(1,max(sz_nt));
for i = 1:1:max(sz_nt)
    [abc, def] = sort(abs(chi_new(i,:)),'descend');
    numb = floor(0.1*combs);
    chill = chi_new(i,def(numb:end));
    neg = max(size(find(chill<0)));
    tot = max(size(chill));
    prob_stable(i) = neg/tot; % Computes cdf less at zero which is the boundary of stability
end