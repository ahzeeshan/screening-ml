%% mp calc data
clear all; close all; clc;

chipres = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (VM-VMc).*(Gs.^2.*(3-4.*nue)+Ge.^2.*(-3+4.*nus)).*k./(2.*z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
chisurf = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) -(VM+VMc).*gamma.*k.^2./2./z;
chitau = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (2.*Ge.*Gs.*(VM + VMc).*(2 -3.*nus + nue.*(-3 + 4.*nus)).*k)./(z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
kcritfn = @(VM,v,z,Gs,Ge,nus,nue,gamma) (Gs.^2.*(-1+v).*(-3+4*nue)-Ge.^2.*(-1+v).*(-3+4*nus)+4.*Ge.*Gs.*(1+v).*(2-3*nus+nue.*(-3+4*nus)))./((1+v) .*gamma .* (Gs.*(-3+4 *nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4*nus)));
nuLi = 0.42;
VLi = 1.3e-5;
GLi = 3.4e9;
gammaLi = 0.556;
ksurf = 2*pi/1e-9;
ksurf=1e8;

%% Train data
d1 = importdata('train-data-posd.mat');
mps = string(d1.mps);
Gtr = d1.Gvrh; Gr = 1e9*d1.Gr; Gv = 1e9*d1.Gv;
Ktr = d1.Kvrh;
%Nu = (3.*K - 2.*G)./(6.*K + 2.*G);
volrat = d1.volratt;

indsbad = unique([find(Gtr<0), find(Gr<0),find(Gv<0)]);
Gtr(indsbad) = NaN;

%%


%chi = chipres(VLi,volrat*VLi,1,G,GLi,nuLi,Nu,ksurf,gammaLi)+chitau(VLi,volrat*VLi,1,G,GLi,nuLi,Nu,ksurf, gammaLi) + chisurf(VLi,volrat*VLi,1,G,GLi,nuLi,Nu,ksurf, gammaLi);
%kcrit = kcritfn(VLi,volrat,1,G,GLi,Nu,nuLi,gammaLi);

%el = find(chi(setdiff(1:end,indsbad))<0)
%mps(el)

%histogram(chi(setdiff(1:end,indsbad))/1e12,'Numbins',10); title('chi')
%figure
%histogram(2*pi*10^(10)./kcrit(setdiff(1:end,indsbad)),'Numbins',50); title('kcrit')

%% Predicted data
d = importdata('non-train-data.mat');

mps = [string(d.mps) ; string(d1.mps)];
G = 1e9*[d.Gvrh, Gtr];
K = 1e9*[d.Kvrh, d1.Kvrh];
volrat = [d.volrat, d1.volratt];

Nu = (3.*K - 2.*G)./(6.*K + 2.*G);

chipreslist = chipres(VLi,volrat*VLi,1,G,GLi,nuLi,Nu,ksurf,gammaLi);
chitaulist = chitau(VLi,volrat*VLi,1,G,GLi,nuLi,Nu,ksurf, gammaLi);
chisurflist = chisurf(VLi,volrat*VLi,1,G,GLi,nuLi,Nu,ksurf, gammaLi);
chi = chipreslist + chitaulist + chisurflist;
kcrit = kcritfn(VLi,volrat,1,G,GLi,Nu,nuLi,gammaLi);
lambdacrit = 2*pi./kcrit;

el = find(chi<0)
mps(el)
find(G<0)

fprintf('Now above 1 nm')
el1 = find(lambdacrit>1e-9)
fprintf('mps:')
mpsnm = mps(el1)
fprintf('lambda:')
lambdacrit(el1)*1e9
fprintf('chi/1e12:')
chinm = chi(el1)/1e12

[lambdasorted, I] = sort(lambdacrit(el1)*1e9,'descend');
mpssorted = mpsnm(I);
chinmsorted = chinm(I);
%%
figure
hold on
ax1=gca;

Gpos0 = @(v,nue,nus) 1./(-1+v)./(-3+4.*nue).*(-4-4.*v+6.*nue+6.*v.*nue+6.*nus+6.*v.*nus-8.*nue.*nus-8.*v.*nue.*nus+sqrt((-1+v).^2.*(-3+4.*nue).*(-3+4.*nus)+4.*(1+v).^2.*(2-3.*nus+nue.*(-3+4*nus)).^2));
fplot(@(v) Gpos0(v,0.42,0.33),[0,1],'k','LineWidth',2);
fplot(@(v) Gpos0(v,0.42,0.5),[0,1],'--k','LineWidth',2);
%
Gnorm = G/GLi;

%scatter(volrat, Gnorm,'d','markerfacecolor','b','markersize',6)

%n = hist3(ax1,[volrat',log(Gnorm)'],'nbins',[10000,1000]);
%pcolor(n)

% [values, centers] = hist3([volrat',log(Gnorm)'],[100 20]);
% imagesc(centers{:}, values.')
% colorbar
% axis xy

%view(2)
%pcolor(n)
%plot(volrat, Gnorm,'d','markerfacecolor','b','markersize',6,'markeredgecolor','b')
scatter(volrat, Gnorm,40,'bd','filled')
set(ax1,'Box','on','yscale','log')
xlim([0,1])
%ylim([-4, 4])
%
%%






set(gcf,'Color','w','Position', [0, 0, 600, 500]);
xlabel(ax1,'$v=V_{\mathrm{Li^+}}/V_{\mathrm{Li}}$','Interpreter','latex','FontWeight','normal','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'$G/G_{\mathrm{Li}}$','Interpreter','latex','FontWeight','normal','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',28,'FontWeight','normal','LineWidth',2,'YTickmode','auto','Fontname','Times New Roman');
%set(gca,'YTick',[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])

annotation('textbox','String','stable','FontSize',28,'Fontname','Times New Roman','EdgeColor','none','Position',[0.67,0.8,0.1,0.1])
annotation('textbox','String','unstable','FontSize',28,'Fontname','Times New Roman','EdgeColor','none','Position',[0.15,0.2,0.1,0.1])
annotation('textbox','String','$\nu_s=0.33$','Interpreter','latex','FontSize',28,'Fontname','Times New Roman','EdgeColor','none','Position',[0.67,0.6,0.1,0.1])
annotation('textbox','String','$\nu_s=0.5$','Interpreter','latex','FontSize',28,'Fontname','Times New Roman','EdgeColor','none','Position',[0.6,0.43,0.1,0.1])
%export_fig -m3 'figs/stabiso.pdf'

%%
figure
histogram(chi(setdiff(1:end,find(G<0)))/1e12,'Numbins',100);
ax1=gca;
xlim([-inf, 150])
set(ax1,'Box','on')
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'$\chi$ /(kJ/mol$\cdot$nm)','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'Number of materials','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',28,'LineWidth',2,'YTickmode','auto','Fontname','Times New Roman')
%xlim([120,200])
%export_fig -m5 chiiso.pdf


%figure
%histogram(kcrit(setdiff(1:end,find(G<0)))); title('kcrit')
figure
histogram(10^9*lambdacrit(setdiff(1:end,find(G<0))),'Numbins',300);
ax1=gca;
xlim([-inf,0.5])
set(ax1,'Box','on')
set(gcf,'Color','w','Position', [0, 0, 600, 500])
xlabel(ax1,'$\lambda_{\mathrm{crit}}$ /nm','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
ylabel(ax1,'Number of materials','Interpreter','latex','FontWeight','bold','FontSize',28,'Fontname','Times New Roman');
set(ax1,'FontName','Arial','FontSize',28,'LineWidth',2,'YTickmode','auto','Fontname','Times New Roman')
%export_fig -m5 lambdaiso.pdf