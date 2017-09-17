function[out] =  check_g_vals_all_element_cubic(C11_matrix, C12_matrix, C44_matrix)
%% This code takes the coefficient matrix and     

    num_ele = max(size(C11_matrix));
    cubic_mat_val = zeros(6,6);
    min_eigs = zeros(num_ele,1);
    out = 0; % default value
    for i = 1:1:num_ele
        C11_mat = C11_matrix(i);
        C12_mat = C12_matrix(i);
        C44_mat = C44_matrix(i);
        cubic_mat_val = [C11_mat ,C12_mat,C12_mat,0      ,0      ,0;...
                         C12_mat ,C11_mat,C12_mat,0      ,0      ,0;...
                         C12_mat ,C12_mat,C11_mat,0      ,0      ,0;...
                        0       ,0      ,0      ,C44_mat,0      ,0;...
                        0       ,0      ,0      ,0      ,C44_mat,0;...
                        0       ,0      ,0      ,0      ,0      ,C44_mat];
        eig_val = eig(cubic_mat_val);
        min_eigs(i) = min(eig_val);

    end
    if min(min_eigs)>0
        out = 1;
    end
end