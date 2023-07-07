% Problem 3 in Chapter 5

clear, close all
% Load data
data_path = "dataLS1.txt";
data = load(data_path);

% Specify X, y for dataLS1 - this data has two columns, first column is X,
% the second column is target. So it's a simple regression.
X = data(:,1);
y = data(:,2);
[N,d] = size(X);

% 3.a 1) Report coefficient rho
% Please calculate the mean of data X and target y as in equation 5.2
% Please calculate the variation of data X and target y as in equation 5.27
cov_xy=cov(X,y)%calculate the covariance matrix
% Please calculate the coefficient as in equation 5.26
rho=cov_xy(1,2)/sqrt(cov_xy(1,1)*cov_xy(2,2))
%the diagonal elements of covariance matrix are the variation of X and y
%and the cov(1,2) or cov(2,1) is the covariance of X&y



fprintf('rho=%.4f\n',rho);

% 3.a 2) Perform linear regression on X, y
x = X;
X=[ones(N,1) X];         % augmented data array
% Compute w, yhat, ybar
w=pinv(X)*y  %the parameter matrix of the linear regression
yhat=X*w  %the estimation of y
ybar=ones(N,1)*mean(y)  %the mean of target y
% Report TSS, ESS, RSS, R-square, and rho
TSS=sum(power(y-ybar,2))
ESS=sum(power(yhat-ybar,2))
RSS=TSS-ESS
R2=ESS/TSS

fprintf('ESS=%.4f\tRSS=%.4f\tTSS=%.4f\tR2=%.4f\n',ESS,RSS,TSS,R2);

% Plot
plot(x,y,'o',x,yhat,'-',x,ybar,'-');
box on
xlim([min(x)-0.2 max(x)+0.2]);
legend({'{\bf y}','$\hat{\bf y}$','$\bar{\bf y}$'},'Interpreter','latex','Location','northwest');
