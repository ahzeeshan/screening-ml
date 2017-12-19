clear all; close all;clc;

%%
lattices = {'cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal-2', 'trigonal-2','triclinic'};

%lattices={'tetragonal-2'};
mps_alltr = []; mps_alltt = []; Gtt = [];Gtr=[]; Gatt = [];Gatr=[];Cmatlist=[];
Cmatttlist = [];Cmattrlist = [];Cmatttlistact = [];Cmattrlistact = [];
Cmatstdttlist = []; Cmatstdtrlist = [];
for i=1:length(lattices)
    disp(lattices(i))
    lattice = lattices{i};
    S = load([lattice,'_posttt_results.mat']);
    Sa = load(fullfile('..','data-gen',[lattice,'-data-posd.mat']));
    S.Gvrhtt = Sa.Gvrh(end-floor(0.1*length(Sa.Gvrh))+1:end);
    S.coeffstest = Sa.ydata(end-floor(0.1*size(Sa.xdata,1))+1:end,:);
    %mps_alltt = [mps_alltt, string(Sa.mps(setdiff(1:end,S.indsbadtt),:))];
    mps_alltt = [mps_alltt; string(Sa.mps(setdiff(1:end,S.indsbadtt),:))];
    Gtt = [Gtt, S.meanGtt(setdiff(1:end,S.indsbadtt))/1e9];
    Gatt = [Gatt, S.Gvrhtt(setdiff(1:end,S.indsbadtt))];
    
    size(S.Gvrhtt(setdiff(1:end,S.indsbadtt)))
    Cmatttlist = [Cmatttlist; S.Cmatrixmeantt(setdiff(1:end,S.indsbadtt),:,:)];
    Cmatstdttlist = [Cmatstdttlist; S.Cmatrixstdtt(setdiff(1:end,S.indsbadtt),:,:)];
    S.Cmatrix_test = []; Ta.Cmatrix_test=[];
    for mat=setdiff(1:length(S.Gvrhtt),S.indsbadtt)
        S.Cmatrix_test(mat,:,:) = constructC(lattice,S.coeffstest(mat,:));
    end
    Cmatttlistact = [Cmatttlistact; S.Cmatrix_test];
    size(Cmatttlistact,1)

    T = load([lattice,'_posttr_results.mat']);
    Ta = load(fullfile('..','data-gen',[lattice,'-data-posd.mat']));
    T.Gvrhtr = Ta.Gvrh(1:end-floor(0.1*length(Ta.Gvrh)));
    T.coeffstr = Ta.ydata(1:end-floor(0.1*length(Ta.Gvrh)),:);
    mps_alltr = [mps_alltr; string(Ta.mps(setdiff(1:end,T.indsbadtr),:))];
    Gtr = [Gtr, T.meanGtr(setdiff(1:end,T.indsbadtr))/1e9];
    Gatr = [Gatr, T.Gvrhtr(setdiff(1:end,T.indsbadtr))];
    
    
    Cmattrlist = [Cmattrlist; T.Cmatrixmeantr(setdiff(1:end,T.indsbadtr),:,:)];
    Cmatstdtrlist = [Cmatstdtrlist; T.Cmatrixstdtr(setdiff(1:end,T.indsbadtr),:,:)];
    for mat=setdiff(1:length(T.Gvrhtr),T.indsbadtr)
        T.Cmatrix_test(mat,:,:) = constructC(lattice,T.coeffstr(mat,:));
    end
    Cmattrlistact = [Cmattrlistact; T.Cmatrix_test];
    size(Cmattrlistact,1)
end

mae(Gtr,Gatr)
mae(Gatt,Gtt)

fx = @(x) x;
figure; hold on;
scatter(Gatt,Gtt, 40,'filled');hold on;
scatter(Gatr,Gtr, 40,'filled');
fplot(fx,'Linewidth',2);title('Test data G','Fontsize',24);
ax1=gca;
axis([1 400 1 400])
set(ax1,'Box','on','xscale','log','yscale','log')
set(gcf,'Color','w','Position', [0, 0, 600, 500]);
xlabel(ax1,'DFT calculated (mp)','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman');
%export_fig('all-lattice-ttG.pdf')

%%
%for mat=1:size(S.coeffstest,1)
%    Sa.Cmatrix_test(mat,:,:) = constructC(lattices{1},S.coeffstest(mat,:));
%end
for i=1:36
    
    if not(all(Cmatttlist(:,i)==0) && all(Cmatttlistact(:,i)==0))
        figure
        %scatter( Cmatrix_test(:,i), Cmatrixmeantt(:,i) ,80,'filled');axis equal;
        %errorbar(Sa.Cmatrix_test(:,i), S.Cmatrixmeantt(:,i), S.Cmatrixstdtt(:,i),'o','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
        errorbar(Cmatttlistact(:,i), Cmatttlist(:,i), Cmatttlist(:,i),'o','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
        hold on;
        fplot(fx,'Linewidth',4)
        title(['test C-',num2str(i)],'Fontsize',24)
        %xlim([0,100]);ylim([0,100]);
        
        ax1=gca;%axis([1 400 1 400])
        set(ax1,'Box','on')
        set(gcf,'Color','w','Position', [0, 0, 600, 500])
        xlabel(ax1,'test actual','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
        ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
        set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
        %%%export_fig([lattice,'-C',num2str(i),'.pdf'])
    end
end

for i=1:36
    
    if not(all(Cmattrlist(:,i)==0) && all(Cmattrlistact(:,i)==0))
        figure
        %scatter( Cmatrix_test(:,i), Cmatrixmeantt(:,i) ,80,'filled');axis equal;
        %errorbar(Sa.Cmatrix_test(:,i), S.Cmatrixmeantt(:,i), S.Cmatrixstdtt(:,i),'o','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
        errorbar(Cmattrlistact(:,i), Cmattrlist(:,i), Cmattrlist(:,i),'o','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
        hold on;
        fplot(fx,'Linewidth',4)
        title(['train C-',num2str(i)],'Fontsize',24)
        %xlim([0,100]);ylim([0,100]);
        
        ax1=gca;%axis([1 400 1 400])
        set(ax1,'Box','on')
        set(gcf,'Color','w','Position', [0, 0, 600, 500])
        xlabel(ax1,'test actual','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
        ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
        set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')
        %%%export_fig([lattice,'-C',num2str(i),'.pdf'])
    end
end

%% Training data

C = load('cubic_posttr_results.mat');
Ca = load(fullfile('..','data-gen','cubic-data-posd.mat'));
Ca.Gvrhtr = Ca.Gvrh(1:end-floor(0.1*length(Ca.Gvrh)));

H = load('hexagonal_posttr_results.mat');
Ha = load(fullfile('..','data-gen','hexagonal-data-posd.mat'));
Ha.Gvrhtr = Ha.Gvrh(1:end-floor(0.1*length(Ha.Gvrh)));

O = load('orthorhombic_posttr_results.mat');
Oa = load(fullfile('..','data-gen','orthorhombic-data-posd.mat'));
Oa.Gvrhtr = Oa.Gvrh(1:end-floor(0.1*length(Oa.Gvrh)));

M = load('monoclinic_posttr_results.mat');
Ma = load(fullfile('..','data-gen','monoclinic-data-posd.mat'));
Ma.Gvrhtr = Ma.Gvrh(1:end-floor(0.1*length(Ma.Gvrh)));

TT = load('tetragonal-2_posttr_results.mat');
TTa = load(fullfile('..','data-gen','tetragonal-2-data-posd.mat'));
TTa.Gvrhtr = TTa.Gvrh(1:end-floor(0.1*length(TTa.Gvrh)));

TR = load('trigonal-2_posttr_results.mat');
TRa = load(fullfile('..','data-gen','trigonal-2-data-posd.mat'));
TRa.Gvrhtr = TRa.Gvrh(1:end-floor(0.1*length(TRa.Gvrh)));

Gtr = [C.meanGtr(setdiff(1:end,C.indsbadtr)), H.meanGtr(setdiff(1:end,H.indsbadtr)), O.meanGtr(setdiff(1:end,O.indsbadtr)), M.meanGtr(setdiff(1:end,M.indsbadtr)), TT.meanGtr(setdiff(1:end,TT.indsbadtr)), TR.meanGtr(setdiff(1:end,TR.indsbadtr))]/1e9;
Gatr = [Ca.Gvrhtr(setdiff(1:end,C.indsbadtr)),Ha.Gvrhtr(setdiff(1:end,H.indsbadtr)),Oa.Gvrhtr(setdiff(1:end,O.indsbadtr)), Ma.Gvrhtr(setdiff(1:end,M.indsbadtr)), TTa.Gvrhtr(setdiff(1:end,TT.indsbadtr)), TRa.Gvrhtr(setdiff(1:end,TR.indsbadtr))];

fx = @(x) x;
figure; hold on;
scatter(Gatr,Gtr, 60,'filled');
fplot(fx,'Linewidth',4);title('Training data G','Fontsize',24);
ax1=gca;
axis([1 400 1 400])
set(ax1,'Box','on','xscale','log','yscale','log')
set(gcf,'Color','w','Position', [0, 0, 600, 500]);
xlabel(ax1,'DFT calculated (mp)','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'NN model predicted','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman');
%export_fig('all-lattice-trG.pdf')

%% kcrit predictions
lattice = 'monoclinic';
lattices = {'cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal-2', 'trigonal-2','triclinic'};

figure
kcritmeanall = [];
%mps_all = [[],[]];
mps_all = "";
for i=1:length(lattices)
    disp(lattices(i))
    lattice = lattices{i};
    S = load([lattice,'_post_results.mat']);
    length(S.indsbad)
    kcritmeanall = [kcritmeanall, S.kcritmean(setdiff(1:end,S.indsbad))];
    size(S.mps_nt(setdiff(1:end,S.indsbad),:))
    %mps_all(end+1:end+size(S.mps_nt(setdiff(1:end,S.indsbad),:),1),:) = [mps_all, S.mps_nt(setdiff(1:end,S.indsbad),:)];
    mps_all = [mps_all;string(S.mps_nt(setdiff(1:end,S.indsbad),:))];
    %mps_all = {cell2mat(mps_all), S.mps_nt(setdiff(1:end,S.indsbad),:)};
end
lambdaall = 2.*pi./kcritmeanall*1e10;

histogram(lambdaall,'NumBins',50); %xlim([0,4.5])
ax1=gca;
set(ax1,'Box','on')
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'$\lambda$ (\AA)','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'\textbf{No. of materials}','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',20,'FontWeight','bold','LineWidth',4,'YTickmode','auto','Fontname','Times New Roman')