function [X,y]=cauchy_data_gp(p,n,beta,gama)
% data generating process
u1=rand(n,1);
u=gama*tan((u1-0.5)*pi);
x1=ones(n,1);
ps=p-1;
for i=1:ps 
    for j=1:ps
        delta(i,j)=0.5^(abs(i-j));
    end
end
x2=mvnrnd(zeros(ps,1),delta,n);
sigma_x=sqrt((1/(n-1))*sum(x2.^2,1));
for i=1:ps 
    x_star(:,i)=x2(:,i)/sigma_x(i);
end
X=[x1 x_star];
y_star=X*beta+u;
y=max(0,y_star);



