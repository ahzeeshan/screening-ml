function [criterion_val]  = getmse(xtrain,ytrain,xtest,ytest)
mdl = fitlm(xtrain, ytrain,'linear','RobustOpts','on');
ypredict = predict(mdl, xtest);
criterion_val = sum((ytest - ypredict).^2);
% criterion_val = mdl.Rsquared.adjusted;
% figure(1);hold on;plot(ytest,'*');
end