clear all

%% Here you should put the predicted valued file with non_training data.
lattice = strtrim(fileread('lattice-type.txt'));
load(strcat(lattice,'_final_results.mat'));

ngoodnets = zeros(1,num_coeffs);
for i=1:num_coeffs
    ngoodnets(i) = length(find(index_out_coeffs{i}==0)) ;
end
ngoodnets