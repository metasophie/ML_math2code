% Problem 3 in Chapter 5

clear, close all
% Load data
data_path = "dataLS2.txt";
data = load(data_path);

% Specify X, y for dataLS2 - this data has two columns, first and second 
% column is X, the second column is target. It's a multiple regression. 
X = data(:,1:2);
y = data(:,3);
[N,d] = size(X);

% Perform linear regression on X, y
x = X;
X=[ones(N,1) X];        % augmented data array
w=pinv(X)*y%求出参数矩阵以及yaht和ybar,进而求出TSS等量
yhat=X*w
ybar=ones(N,1)*mean(y)

% Report TSS, ESS, RSS, R-square, and rho
TSS=sum(power(y-ybar,2))
ESS=sum(power(yhat-ybar,2))
RSS=TSS-ESS
R2=ESS/TSS

fprintf('ESS=%.4f\tRSS=%.4f\tTSS=%.4f\tR2=%.4f\n',ESS,RSS,TSS,R2);

% Plot function from Prof.Wang
a=sym('x',[2 1]);
b=sym('w',[3 1]);
f=@(a,b)b(1)+b(2)*a(1)+b(3)*a(2);

% Plot the regression results
figure, hold on
myPlot3D(x,w,f);
plot3(x(:,1),x(:,2),y,'o')


function myPlot3D(X,w,f)
    xmin=min(X(:,1));    xmax=max(X(:,1));
    ymin=min(X(:,2));    ymax=max(X(:,2));
    [X, Y]=meshgrid(xmin:0.05:xmax, ymin:0.05:ymax);
    m=size(X,2);
    n=size(Y,1);
    Z=zeros(m,n);
    x=zeros(2,1);
    for i=1:m
        x(1)=X(1,i);
        for j=1:n
            x(2)=Y(j,1);
            Z(i,j)=f(x,w); 
        end
    end
    surf(X,Y,Z')
    grid on
    view(3)
end