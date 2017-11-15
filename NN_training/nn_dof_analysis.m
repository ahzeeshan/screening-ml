clear all;close all;clc;

lattice = {'cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal-2', 'trigonal-2'};

lattice = 'tetragonal-2';
load( fullfile( '..','data-gen',strcat(lattice,'-data.mat') ) );
load(fullfile('..','data-gen',strcat(lattice,'-data.mat'))) % xdata and ydata
load(fullfile('..','Linear',strcat('features_',lattice,'.mat')));
load(fullfile('..','data-gen',strcat(lattice,'-non-training-data.mat')));
X_mat = xdata(1:end-floor(0.1*size(xdata,1)),:);
disp(lattice)
size(X_mat)
max_neurons = zeros(1,1);
for i=1:size(ydata,2)
    %disp(i)
    %length(feature_list{i})
    %size(X_mat,1)/(2*length(feature_list{i}))
    %(size(X_mat,1)-2)/(length(feature_list{i})+1)
    disp([i, length(feature_list{i}), (size(X_mat,1)-1)/(length(feature_list{i})+2)])
end

