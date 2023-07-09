% Problem 4 in Chapter 5

clear, close all
% Set random seed
rng(2023);

% Generate data
x1 = [2.4, 3.1, 3.8, 2.3, 2.0, 3.7, 3.2, 3.0, 2.8]';
y = [4.6, 6.1, 7.7, 4.9, 4.1, 7.4, 6.3, 5.8, 5.5]';
y_mean = mean(y);

N = size(x1,1);
x1_mean = mean(x1);

% Specify a like 1, 10, 100, 1000
a = 100;
x2 = x1 + a*rand(N,1);
x2_mean = mean(x2);

% Calculate coefficient of x1 and x2
cov_x=cov(x1,x2)
rho=cov_x(1,2)/sqrt(cov_x(1,1)*cov_x(2,2))

fprintf('Coefficient: %.4f\n', rho);

% Calculate eigenvalues of XX^T to see how ill-conditioned the problem is
X = [x1,x2];
e = eig(X'*X);
fprintf('Eigenvalues are %.16f, %.16f\n', e(1), e(2));

% Perform ridge regression on X, y
% Report R-square
% Specify lambda
lambda = 0.1;
old_x = X;
X=[ones(N,1) X];        % augmented data array
d = size(X,2);
w=inv(X'*X+lambda*eye(3))*X'*y  %add a 3*3 identity matrix
yhat=X*w                        %calculate yhat&ybar and Rsquare
ybar=ones(N,1)*y_mean
TSS=sum(power(y-ybar,2))
ESS=sum(power(yhat-ybar,2))
R2=ESS/TSS

fprintf('R-square is %.4f\n', R2);