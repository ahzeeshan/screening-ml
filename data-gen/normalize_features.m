clear all
close all
clc

a = importdata('featureswoNaN.txt');

for i=1:size(a,2)
    b(:,i) = (a(:,i)-min(a(:,i))) / (max(a(:,i))-min(a(:,i))) ;
end

b = b(:,[1:12,14:16,19:21]);
dlmwrite('normalized-features.txt',b,'delimiter','\t')