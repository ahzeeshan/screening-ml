clear all; close all; clc;

lattice = 'cubic';

load(fullfile('..','data-gen',strcat(lattice,'-data-posd.mat')));
X_mat = xdata;
%xtest = xdata(end-floor(0.1*size(xdata,1))+1:end,:);
coeffs = ydata;
%coeffstest = ydata(end-floor(0.1*size(xdata,1))+1:end,:);

sz_tot = size(X_mat,1);
%sz_tt = size(xtest,1);
is_posdef = zeros(1,sz_tot);
for mat=1:sz_tot
    Cmat = constructC(lattice,coeffs(mat,:));
    [~,p] = chol(Cmat);
    if (det(Cmat)>0)
        [~, p] = chol(Cmat);
        if(p==0)
            is_posdef(mat) = 1;
        end
    end
end
% 
% for mat=sz_tr+1:sz_tt+sz_tr
%     Cmat = constructC(lattice,coeffstest(mat,:));
%     [~,p] = chol(Cmat);
%     if (det(Cmat)>0)
%         [~, p] = chol(Cmat);
%         if(p==0)
%             is_posdef(mat) = 1;
%         end
%     end
% end
mps(find(is_posdef==0),:)