clear all

%% Here you should put the predicted valued file with non_training data.
lattice = strtrim(fileread('lattice-type.txt'));
load(fullfile('..','data-gen',strcat(lattice,'-non-training-data.mat')));
mps_nt = mps; clear mps;
load(fullfile('..','data-gen',strcat('pred-',lattice,'-non-training-data.mat')));
G_mp = G; K_mp = K; clear G K;
load(fullfile('..','Linear',strcat('features_',lattice,'.mat')));
load(strcat(lattice,'_final_results.mat'));
load(fullfile('..','data-gen',strcat(lattice,'-data-posd.mat')));

disp(lattice)
num_samples = 1000;
X_mat = xdata(trainIndglob,:);
coeffs = ydata(trainIndglob,:);
Gvrhtr = Gvrh(trainIndglob);
cubic_nt = xntdata;

% test data
xtest = xdata(testIndglob,:);
coeffstest = ydata(testIndglob,:);
Gvrhtt = Gvrh(testIndglob);

calcRsq = @(predvals,actualvals) 1 - sum((predvals - actualvals).^2)/sum((actualvals - mean(actualvals)).^2);
fx=@(x) x;
%%
sz_tr = size(X_mat,1);
sz_nt = size(cubic_nt,1);
sz_tt = size(xtest,1);
num_coeffs = size(coeffs,2);
tr_perf = zeros(num_coeffs, num_samples);
Coeffs_cell = cell(num_samples, num_coeffs);
predcoeffs = zeros(sz_nt, num_coeffs, num_samples);
predcoeffs_test = zeros(sz_tt, num_coeffs, num_samples);
predcoeffs_tr = zeros(sz_tr, num_coeffs, num_samples);
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
        %tr_perf(coeff_num,num_samp) = tr.perf(end);
        
        Coeffs_tr = net(x);
        Coeffs_mat = net(non_training);
        Coeffs_test_mat = net(xtest_mapped);
        Coeffs_mat = mapminmax('reverse',Coeffs_mat,t_recover);
        Coeffs_test_mat = mapminmax('reverse',Coeffs_test_mat,t_recover);
        Coeffs_tr = mapminmax('reverse',Coeffs_tr,t_recover);
        predcoeffs_tr(:,coeff_num,num_samp) = Coeffs_tr;
        predcoeffs(:,coeff_num, num_samp) = Coeffs_mat;
        predcoeffs_test(:, coeff_num, num_samp) = Coeffs_test_mat;
        
        tr_perf(coeff_num,num_samp) = calcRsq(predcoeffs_tr(:,coeff_num,num_samp),coeffs(:,coeff_num));
        %1 - sum((coeffs(:,coeff_num) - predcoeffs_tr(:,coeff_num,num_samp)).^2)/sum((coeffs(:,coeff_num) - mean(coeffs(:,coeff_num))).^2);
        test_perf(coeff_num, num_samp) = calcRsq(predcoeffs_test(:,coeff_num,num_samp),coeffstest(:,coeff_num));
        %1 - sum((coeffstest(:,coeff_num) - predcoeffs_test(:,coeff_num,num_samp)).^2)/sum((coeffstest(:,coeff_num) - mean(coeffstest(:,coeff_num))).^2);
    end
end


%% generating model choices and good nets
ngoodnets = zeros(1,num_coeffs);
ngoodnets_tr = zeros(1,num_coeffs);
ngoodnets_test = zeros(1,num_coeffs);
sortedtestperf = cell(1,num_coeffs); sortedtrperf = cell(1,num_coeffs);
sortind_tr = cell(1, num_coeffs); sortind_test = cell(1, num_coeffs);
for i=1:num_coeffs
    %ngoodnets_tr(i) = length(find(index_out_coeffs{i}==0));
    ngoodnets_tr(i) = length(find(tr_perf(i,:)>0.75));
    ngoodnets_test(i) = length(find(test_perf(i,:)>0.5));
    if(ngoodnets_tr(i)==0)
        ngoodnets_tr(i) = 100;
    end
    ngoodnets(i) = min(ngoodnets_test(i),ngoodnets_tr(i));
    [sortedtestperf{i},sortind_test{i}] = sort(test_perf(i,:),'descend');
    [sortedtrperf{i},sortind_tr{i}] = sort(tr_perf(i,:),'descend');
    %ngoodnets(i) = 1;
end
%ngoodnets_tr=1000*ones(1,num_coeffs)
%if any(not(ngoodnets_tr))
%    ngoodnets_tr=ones(1,num_coeffs);
%end
ngoodnets_tr
ngoodnets_test
ngoodnets

%% plotting parity plot against training data

for i=1:num_coeffs
    figure
    scatter( coeffs(:,i), predcoeffs_tr(:,i,sortind_tr{i}(2)) ,80,'filled');axis equal;
    hold on;
    fplot(fx,'Linewidth',4)
    title(['C-',num2str(i)],'Fontsize',24)
    %xlim([0,100]);ylim([0,100]);
    ax1=gca;
    set(ax1,'Box','on')
    set(gcf,'Color','w','Position', [0, 0, 600, 500])
    xlabel(ax1,'train actual','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
    ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
    set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
    %%%export_fig([lattice,'-tr-C',num2str(i),'.pdf'])
end

%% plotting parity plot against test data for the best case
fx=@(x) x;
for i=1:num_coeffs
    figure
    scatter( coeffstest(:,i), predcoeffs_test(:,i,sortind_tr{i}(1)) ,80,'filled');axis equal;
    hold on;
    fplot(fx,'Linewidth',4)
    title(['C-',num2str(i)],'Fontsize',24)
    %xlim([0,100]);ylim([0,100]);
    ax1=gca;
    set(ax1,'Box','on')
    set(gcf,'Color','w','Position', [0, 0, 600, 500])
    xlabel(ax1,'test actual','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
    ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
    set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
    %%%export_fig([lattice,'-C',num2str(i),'.pdf'])
end


%%
combs = 10000;
%cubic_mat = zeros(6,6,sz_nt,combs); -- not storing right now
% G_v = zeros(sz_nt,combs);
% G_r = zeros(sz_nt,combs);
% B_v = zeros(sz_nt,combs);
% B_r = zeros(sz_nt,combs);
% Nu = zeros(sz_nt,combs);
% G = zeros(sz_nt,combs);
% B = zeros(sz_nt,combs);
nue = 0.42;
VM = 1.3e-05;
Ge = 3.4e09;
gamma = 0.556;
k=5*10^10;


chipres = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (VM-VMc).*(Gs.^2.*(3-4.*nue)+Ge.^2.*(-3+4.*nus)).*k./(2.*z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
chisurf = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) -(VM+VMc).*gamma.*k.^2./2./z;
chitau = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (2.*Ge.*Gs.*(VM + VMc).*(2 -3.*nus + nue.*(-3 + 4.*nus)).*k)./(z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
kcritfn = @(VM,v,z,Gs,Ge,nus,nue,gamma) 2*(-1+v)*(Gs^2*(-3+4*nue)+Ge^2*(3-4*nus))/((1+v)*gamma)/(Gs*(-3+4*nue)*(-1+nus)+Ge*(-1+nue)*(-3+4*nus));

model_choice = zeros(num_coeffs, combs);
for i=1:num_coeffs
    model_choice(i,:) = datasample( sortind_tr{i}(1:ngoodnets_tr(i)) , combs);
end

%% Training data from Model
for i=1:num_coeffs
    model_choice(i,:) = datasample( sortind_tr{i}(1:ngoodnets_tr(i)) , combs);
end
Gnewtr = cell(1,sz_tr);
Nunewtr = cell(1,sz_tr);
ind_posdefntr = cell(1,sz_tr);
Cmatrixtr = cell(sz_tr,combs);
G = zeros(sz_tr,combs);
B = zeros(sz_tr,combs);
Nu = zeros(sz_tr,combs);
is_posdef = zeros(sz_tr, combs);
chi_new = zeros(sz_tr,combs);
kcrit = zeros(sz_tr,combs);
for mat=1:sz_tr
    mat
    for i=1:combs
        coeffs_modeltr = [];
        for j=1:num_coeffs
            coeffs_modeltr = [coeffs_modeltr, predcoeffs_tr(mat, j, model_choice(j,i))];
        end
        Cmat = constructC(lattice, coeffs_modeltr);
        Cmatrixtr{mat,i} = Cmat;
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
        chi = chipres(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),k,gamma)+chitau(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),k, gamma) + 1*chisurf(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),k, gamma);
        chi_new(mat,i) = chi./10^12;
        kcrit(mat,i) = kcritfn(VM,vol_new(mat),1,G(mat,i),Ge,Nu(mat,i),nue,gamma);
    end
    ind_posdefntr{mat} = find(is_posdef(mat,:)==1);
    Gnewtr{mat} = G(mat,ind_posdefntr{mat});
    Nunewtr{mat} = Nu(mat,ind_posdefntr{mat});
end
prob_stabletr = zeros(1,sz_tr);
meanGtr = zeros(1,sz_tr);
stdGtr = zeros(1,sz_tr);
meanNutr = zeros(1,sz_tr);
stdNutr = zeros(1,sz_tr);
kcritmeantr = zeros(1,sz_tr);
chimeantr = zeros(1,sz_tr);
Cmatrixmeantr = zeros(sz_tr,6,6);
Cmatrixstdtr = zeros(sz_tr,6,6);
indsbadtr = [];
for mat = 1:sz_tr
    if ~isempty(ind_posdefntr{mat})
        neg = size(find(chi_new(mat,ind_posdefntr{mat})<0),2);
        tot = size(chi_new(mat,ind_posdefntr{mat}),2);
        prob_stabletr(mat) = neg/tot;
        meanGtr(mat) = mean(Gnewtr{mat});
        stdGtr(mat) = std(Gnewtr{mat});
        meanNutr(mat) = mean(Nunewtr{mat});
        stdNutr(mat) = std(Nunewtr{mat});
        chimeantr(mat) = mean(chi_new(mat,ind_posdefntr{mat}));
        kcritmeantr(mat) = mean(kcrit(mat,ind_posdefntr{mat}));
        D = cat(3,Cmatrixtr{mat,ind_posdefntr{mat}});
        Cmatrixmeantr(mat,:,:) = mean(D,3);
        Cmatrixstdtr(mat,:,:) = std(D,[],3);
    else
        indsbadtr = [indsbadtr, mat];
        prob_stabletr(mat) = NaN;
        meanGtr(mat) = NaN;
        stdGtr(mat) = NaN;
        meanNutr(mat) = NaN;
        stdNutr(mat) = NaN;
        chimeantr(mat) = NaN;
        kcritmeantr(mat) = NaN;
        D = cat(3,Cmatrixtr{mat,ind_posdefntr{mat}});
        Cmatrixmeantr(mat,:,:) = NaN;
        Cmatrixstdtr(mat,:,:) = NaN;
    end
end


Rsq_trG = calcRsq(meanGtr(setdiff(1:end,indsbadtr))/10^9, Gvrhtr(setdiff(1:end,indsbadtr)));
Rsq_trG
save(strcat(lattice,'_posttr_results.mat'),'mps','meanGtr','stdGtr', 'indsbadtr','Cmatrixmeantr','Cmatrixstdtr');


figure; hold on;
scatter(Gvrhtr(setdiff(1:end,indsbadtr)),meanGtr(setdiff(1:end,indsbadtr))/10^9, 60,'filled');
fplot(fx,'Linewidth',4);title('Training data G','Fontsize',24);
ax1=gca;
%axis([1 400 1 400]);
set(ax1,'Box','on')
%set(ax1,'xscale','log','yscale','log');
set(gcf,'Color','w','Position', [0, 0, 600, 500]);
xlabel(ax1,'DFT calculated (mp)','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman');
%export_fig([lattice,'-trG.pdf'])
%% Test data shear modulus
for i=1:num_coeffs
    model_choice(i,:) = datasample( sortind_tr{i}(1:ngoodnets_tr(i)) , combs);
end
%Cmat_tt = cell(1,sz_tt);
Gnewtt = cell(1,sz_tt);
Nunewtt = cell(1,sz_tt);
ind_posdefntt = cell(1,sz_tt);
Cmatrixtt = cell(sz_tt,combs);
G = zeros(sz_tt,combs);
B = zeros(sz_tt,combs);
Nu = zeros(sz_tt,combs);
is_posdef = zeros(sz_tt, combs);
chi_new = zeros(sz_tt,combs);
Cmatrix_test = zeros(sz_tt,6,6);
kcrit = zeros(sz_tt,combs);
for mat=1:sz_tt
    mat
    Cmatrix_test(mat,:,:) = constructC(lattice,coeffstest(mat,:));
    for i=1:combs
        coeffs_model = [];
        for j=1:num_coeffs
            coeffs_model = [coeffs_model, predcoeffs_test(mat, j, model_choice(j,i))];
        end
        Cmat = constructC(lattice, coeffs_model);
        
        %Cmat_tt = constructC(lattice, coeffstest(mat,:));
        Cmatrixtt{mat,i} = Cmat;
        %Cmatrix{mat,i} = Cmat;
        [~,p] = chol(Cmat);
        if (det(Cmat)~=0)
            [~, p] = chol(Cmat);
            if(p==0)
                is_posdef(mat,i) = 1;
            end
            Smat = inv(Cmat);
            %Smat_tt = inv(Cmat_tt);
            G_v = (Cmat(1,1)+Cmat(2,2)+Cmat(3,3)+3*(Cmat(4,4)+Cmat(5,5)+Cmat(6,6))-Cmat(1,2)-Cmat(2,3)-Cmat(3,1))/15;
            B_v = (Cmat(1,1)+Cmat(2,2)+Cmat(3,3)+2*(Cmat(1,2)+Cmat(2,3)+Cmat(3,1)))/9;
            B_r = 1/(Smat(1,1)+Smat(2,2)+Smat(3,3)+2*(Smat(1,2)+Smat(2,3)+Smat(3,1)));
            G_r = 15/(4*(Smat(1,1)+Smat(2,2)+Smat(3,3))+3*(Smat(4,4)+Smat(5,5)+Smat(6,6))-4*(Smat(1,2)+Smat(2,3)+Smat(3,1)));
            
            %G_vtt = (Cmat_tt(1,1)+Cmat_tt(2,2)+Cmat_tt(3,3)+3*(Cmat_tt(4,4)+Cmat_tt(5,5)+Cmat_tt(6,6))-Cmat_tt(1,2)-Cmat_tt(2,3)-Cmat_tt(3,1))/15;
            %B_vtt = (Cmat_tt(1,1)+Cmat_tt(2,2)+Cmat_tt(3,3)+2*(Cmat_tt(1,2)+Cmat_tt(2,3)+Cmat_tt(3,1)))/9;
            %B_rtt = 1/(Smat_tt(1,1)+Smat_tt(2,2)+Smat_tt(3,3)+2*(Smat_tt(1,2)+Smat_tt(2,3)+Smat_tt(3,1)));
            %G_rtt = 15/(4*(Smat_tt(1,1)+Smat_tt(2,2)+Smat_tt(3,3))+3*(Smat_tt(4,4)+Smat_tt(5,5)+Smat_tt(6,6))-4*(Smat_tt(1,2)+Smat_tt(2,3)+Smat_tt(3,1)));
            
        else
            G_v = 0;
            G_r = 0;
            B_v = 0;
            B_r = 0;
        end
        
        B(mat,i) = 0.5*(B_v+B_r)*10^9;
        G(mat,i) = 0.5*(G_v+G_r)*10^9;
        %Btt(mat) = 0.5*(B_vtt+B_rtt)*10^9;
        %Gtt(mat) = 0.5*(G_vtt+G_rtt)*10^9;
        
        %ind_new = (1:sz_nt);
        Nu(mat, i) = (3.*B(mat, i) - 2.*G(mat, i))./(6.*B(mat, i) + 2.*G(mat, i));
        %vol_new = volrat(ind_new);
        %chi = chipres(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),k,gamma)+chitau(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),k, gamma) + 1*chisurf(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),k, gamma);
        chi_new(mat,i) = chi./10^12;
        kcrit(mat,i) = kcritfn(VM,vol_new(mat),1,G(mat,i),Ge,Nu(mat,i),nue,gamma);
    end
    ind_posdefntt{mat} = find(is_posdef(mat,:)==1);
    Gnewtt{mat} = G(mat,ind_posdefntt{mat});
    %meanGtest(mat) = mean(Gnew{mat});
    %stdGtest(mat) = std(Gnew{mat});
    Nunew{mat} = Nu(mat,ind_posdefntt{mat});
end

prob_stable = zeros(1,sz_tt);
meanGtt = zeros(1,sz_tt);
stdGtt = zeros(1,sz_tt);
meanNutt = zeros(1,sz_tt);
stdNutt = zeros(1,sz_tt);
kcritmeantt = zeros(1,sz_tt);
chimeantt = zeros(1,sz_tt);
Cmatrixmeantt = zeros(sz_tt,6,6);
Cmatrixstdtt = zeros(sz_tt,6,6);
indsbadtt = [];
for mat = 1:sz_tt
    if ~isempty(ind_posdefntt{mat})
        neg = size(find(chi_new(mat,ind_posdefntt{mat})<0),2);
        tot = size(chi_new(mat,ind_posdefntt{mat}),2);
        prob_stable(mat) = neg/tot;
        meanGtt(mat) = mean(Gnewtt{mat});
        stdGtt(mat) = std(Gnewtt{mat});
        meanNutt(mat) = mean(Nunew{mat});
        stdNutt(mat) = std(Nunew{mat});
        chimeantt(mat) = mean(chi_new(mat,ind_posdefntt{mat}));
        kcritmeantt(mat) = mean(kcrit(mat,ind_posdefntt{mat}));
        D = cat(3,Cmatrixtt{mat,ind_posdefntt{mat}});
        Cmatrixmeantt(mat,:,:) = mean(D,3);
        Cmatrixstdtt(mat,:,:) = std(D,[],3);
    else
        indsbadtt = [indsbadtt, mat];
    end
end
Rsq_ttG = calcRsq(meanGtt(setdiff(1:end,indsbadtt))/10^9, Gvrhtt(setdiff(1:end,indsbadtt)))
save(strcat(lattice,'_posttt_results.mat'),'mps','meanGtt','stdGtt', 'indsbadtt','Cmatrixmeantt','Cmatrixstdtt');

%scatter(Gtt/10^9, meanGtest/10^9, 60,'filled'); hold on;
%fplot(fx,'Linewidth',4)
figure;hold on;
scatter(Gvrhtt(setdiff(1:end,indsbadtt)),meanGtt(setdiff(1:end,indsbadtt))/10^9, 60,'filled')
errorbar(Gvrhtt(setdiff(1:end,indsbadtt)), meanGtt(setdiff(1:end,indsbadtt))/10^9, stdGtt(setdiff(1:end,indsbadtt))/10^9,'o','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
fplot(fx,'Linewidth',4);title('Test data G','Fontsize',24)
%xlim([0,120]);ylim([0,120]);
ax1=gca;
axis([1 400 1 400])
set(ax1,'Box','on','xscale','log','yscale','log')
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'mp-predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
%export_fig([lattice,'-ttG.pdf'])

for i=1:21
    figure
    %scatter( Cmatrix_test(:,i), Cmatrixmeantt(:,i) ,80,'filled');axis equal;
    errorbar(Cmatrix_test(:,i), Cmatrixmeantt(:,i), Cmatrixstdtt(:,i),'o','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
    
    hold on;
    fplot(fx,'Linewidth',4)
    title(['C-',num2str(i)],'Fontsize',24)
    %xlim([0,100]);ylim([0,100]);
    ax1=gca;
    set(ax1,'Box','on')
    set(gcf,'Color','w','Position', [0, 0, 600, 500])
    xlabel(ax1,'test actual','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
    ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
    set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
    %%%export_fig([lattice,'-C',num2str(i),'.pdf'])
end


%% Predictions on nt data
for i=1:num_coeffs
    model_choice(i,:) = datasample( sortind_tr{i}(1:ngoodnets_tr(i)) , combs);
end
is_posdef = zeros(sz_nt, combs);
chi_new = zeros(sz_nt,combs);
kcrit = zeros(sz_nt,combs);
Gnew = cell(1,sz_nt);
Nunew = cell(1,sz_nt);
ind_posdefn = cell(1,sz_nt);
Cmatrix = cell(sz_nt,combs);
G = zeros(sz_nt,combs);
B = zeros(sz_nt,combs);
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
        chi = chipres(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),k,gamma)+chitau(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),k, gamma) + 1*chisurf(VM,vol_new(mat)*VM,1,G(mat,i),Ge,nue,Nu(mat,i),k, gamma);
        chi_new(mat,i) = chi./10^12;
        kcrit(mat,i) = kcritfn(VM,vol_new(mat),1,G(mat,i),Ge,Nu(mat,i),nue,gamma);
    end
    ind_posdefn{mat} = find(is_posdef(mat,:)==1);
    Gnew{mat} = G(mat,ind_posdefn{mat});
    Nunew{mat} = Nu(mat,ind_posdefn{mat});
end

prob_stable = zeros(1,sz_nt);
meanG = zeros(1,sz_nt);
stdG = zeros(1,sz_nt);
meanNu = zeros(1,sz_nt);
stdNu = zeros(1,sz_nt);
kcritmean = zeros(1,sz_nt);
chimean = zeros(1,sz_nt);
Cmatrixmean = zeros(sz_nt,6,6);
Cmatrixstd = zeros(sz_nt,6,6);
indsbad = [];
for mat = 1:sz_nt
    if ~isempty(ind_posdefn{mat})
        neg = size(find(chi_new(mat,ind_posdefn{mat})<0),2);
        tot = size(chi_new(mat,ind_posdefn{mat}),2);
        prob_stable(mat) = neg/tot;
        meanG(mat) = mean(Gnew{mat});
        stdG(mat) = std(Gnew{mat});
        meanNu(mat) = mean(Nunew{mat});
        stdNu(mat) = std(Nunew{mat});
        chimean(mat) = mean(chi_new(mat,ind_posdefn{mat}));
        kcritmean(mat) = mean(kcrit(mat,ind_posdefn{mat}));
        D = cat(3,Cmatrix{mat,ind_posdefn{mat}});
        Cmatrixmean(mat,:,:) = mean(D,3);
        Cmatrixstd(mat,:,:) = std(D,[],3);
    else
        indsbad = [indsbad, mat];
    end
end
save(strcat(lattice,'_post_results.mat'),'mps_nt','chimean', 'meanG', 'kcritmean','prob_stable', 'stdG', 'ngoodnets','ind_posdefn','Cmatrixmean','Cmatrixstd','indsbad','-v7.3');

%% Materials Project prediction Comparison
figure
%meanG(indsbad)=NaN;
%scatter(G_mp,meanG/10^9,'o')
scatter(G_mp(setdiff(1:end,indsbad)), meanG(setdiff(1:end,indsbad))/10^9, 60,'filled'); hold on;
fx=@(x) x;
fplot(fx,'Linewidth',4)
xlim([1,400]);ylim([1,400]);
ax1=gca;
set(ax1,'Box','on','xscale','log','yscale','log')
%axis([0 400 0 400])
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'mp-predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
export_fig([lattice,'-G_mp.pdf'])

%% Materials Project comparison with error bars
figure
errorbar(G_mp(setdiff(1:end,indsbad)), meanG(setdiff(1:end,indsbad))/10^9, stdG(setdiff(1:end,indsbad))/10^9,'o','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
fplot(fx,'Linewidth',4);
xlim([1,400]);ylim([1,400]);
ax1=gca;
set(ax1,'Box','on','xscale','log','yscale','log')
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'mp-predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
%export_fig([lattice,'-GwErr_mp.pdf'])




%% Probability of stability for all materials
figure
%prob_stable(indsbad)=NaN;
% histogram(meanG/10^9,'NumBins',25)
histogram(prob_stable(setdiff(1:end,indsbad)),12)
ax1=gca;
set(ax1,'Box','on')
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'\textbf{Probability of Stability}','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'\textbf{No. of materials}','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
%
%% Plotting chi for all materials
%chimean(indsbad)=NaN;
figure
histogram(chimean(setdiff(1:end,indsbad)),25)
ax1=gca;
set(ax1,'Box','on')
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'\textbf{Stability parameter (kJ/mol.nm)}','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'\textbf{No. of materials}','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
%xlim([120,200])

% %%
% figure
% mat=3;
% histogram(chi_new(mat,ind_posdefn{mat}),25)
% ax1=gca;
% set(ax1,'Box','on')
% set(gcf,'Color','w','Position', [0, 0, 600, 500])
% xlabel(ax1,'\textbf{Stability parameter (kJ/mol.nm)}','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
% ylabel(ax1,'\textbf{No. of Neural Networks}','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
% set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')

%% Plotting critical k for all materials
%kcritmean(indsbad) = NaN;
figure
histogram(kcritmean(setdiff(1:end,indsbad)),25)
ax1=gca;
set(ax1,'Box','on')
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'$k_{crit}$','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'\textbf{No. of materials}','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
export_fig([lattice,'-kcritm_mp.pdf'])
