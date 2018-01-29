%% mp calc data
clear all; close all; clc;

chipres = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (VM-VMc).*(Gs.^2.*(3-4.*nue)+Ge.^2.*(-3+4.*nus)).*k./(2.*z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
chisurf = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) -(VM+VMc).*gamma.*k.^2./2./z;
chitau = @(VM,VMc,z,Gs,Ge,nue,nus,k,gamma) (2.*Ge.*Gs.*(VM + VMc).*(2 -3.*nus + nue.*(-3 + 4.*nus)).*k)./(z.*(Gs.*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus)));
kcritfn = @(VM,v,z,Gs,Ge,nus,nue,gamma) 2.*(-1+v).*(Gs.^2*(-3+4.*nue)+Ge.^2*(3-4.*nus))./((1+v).*gamma)./(Gs*(-3+4.*nue).*(-1+nus)+Ge.*(-1+nue).*(-3+4.*nus));
nue = 0.42;
VM = 1.3e-05;
Ge = 3.4e09;
gamma = 0.556;
k=1e8;

%%
d=importdata('train-data-posd.mat');
mps = string(d.mps);
G = 1e9*d.Gvrh; Gr = 1e9*d.Gr; Gv = 1e9*d.Gv;
K = 1e9*d.Kvrh;
Nu = (3.*K - 2.*G)./(6.*K + 2.*G);
volrat = d.volratt;

chi = chipres(VM,volrat*VM,1,G,Ge,nue,Nu,k,gamma)+chitau(VM,volrat*VM,1,G,Ge,nue,Nu,k, gamma) + 1*chisurf(VM,volrat*VM,1,G,Ge,nue,Nu,k, gamma);
kcrit = kcritfn(VM,volrat,1,G,Ge,Nu,nue,gamma);

indsbad = unique([find(G<0), find(Gr<0),find(Gv<0)])
el = find(chi(setdiff(1:end,indsbad))<0)
mps(el)

histogram(chi(setdiff(1:end,indsbad))); title('chi')
figure
histogram(kcrit(setdiff(1:end,indsbad))); title('kcrit')

%% Predicted data
d=importdata('non-train-data.mat');
mps = string(d.mps);
G = 1e9*d.Gvrh;
K = 1e9*d.Kvrh;
Nu = (3.*K - 2.*G)./(6.*K + 2.*G);
volrat = d.volrat;

chi = chipres(VM,volrat*VM,1,G,Ge,nue,Nu,k,gamma)+chitau(VM,volrat*VM,1,G,Ge,nue,Nu,k, gamma) + 1*chisurf(VM,volrat*VM,1,G,Ge,nue,Nu,k, gamma);
kcrit = kcritfn(VM,volrat,1,G,Ge,Nu,nue,gamma);

el = find(chi<0)
mps(el)
find(G<0)

histogram(chi(setdiff(1:end,find(G<0)))); title('chi')
figure
histogram(kcrit(setdiff(1:end,find(G<0)))); title('kcrit')