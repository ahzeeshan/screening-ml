clear all
%close all
%clc
tic
%% stepwisefit
% lattice = {'cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal-2', 'trigonal-2'};
% lattice = {'trigonal-2'};
% for i=1:length(lattice)
% fprintf(lattice{i})
% fprintf('\n')
% load( fullfile( '..','data-gen',strcat(lattice{i},'-data.mat') ) )
% for j=1:size(ydata,2)
%     [b,se,pval,inmodel,stats,nextstep,history] = stepwisefit(xdata, ydata(:,j),'inmodel',[1:9, 10, 11,12, 13:18],'scale','on','penter',0.03,'premove',0.15);
% end
% end

%% sequential
lattice = 'cubic';
warning('off')
load( fullfile('..','data-gen',strcat(lattice,'-data-posd.mat')));
xdata = xdata(1:end-floor(0.1*size(xdata,1)),:);
ydata = ydata(1:end-floor(0.1*size(xdata,1)),:);
n=100;
feature_list = cell(1,size(ydata,2));
inmodel = cell(1,n);
history = cell(1,n);
for ncoeff=1:size(ydata,2)
criteria = 100000000;
for i=1:n
load( fullfile('..','data-gen',strcat(lattice,'-data.mat')));
opts = statset('display','iter','TolTypeFun','rel');
[inmodel{i},history{i}] = sequentialfs(@getmse,xdata,ydata(:,ncoeff),'cv',5,'options',opts,'direction','backward');
if criteria > history{i}.Crit(end)
    criteria = history{i}.Crit(end);
    min_index = i;
end
end
criteria_list(ncoeff) = criteria;
a = (1:size(xdata,2));
feature_list{ncoeff} = a(inmodel{min_index});
end
save(strcat('features_',lattice,'.mat'), 'feature_list','criteria_list')
toc