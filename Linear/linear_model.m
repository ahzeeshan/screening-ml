clear all
close all
clc

load(fullfile('..','data-gen','cubic-data.mat'))
for i=1:size(ydata,2)
    
    for 
    mdl = fitlm(xdata, ydata(:,i));
    
    
end

