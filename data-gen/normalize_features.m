clear all
close all
clc

a = importdata('featureswoNaN.txt');

for i=1:size(a,2)
    a(:,i) = (a(:,i)-min(a(:,i))) / (max(a(:,i))-min(a(:,i))) ;
end

a = a(:,[1:12,14:16,19:20]);
dlmwrite('normalized-features.txt',a,'delimiter','\t')