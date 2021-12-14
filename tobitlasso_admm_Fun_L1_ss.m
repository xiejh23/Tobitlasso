function beta1=tobitlasso_admm_Fun_L1_ss(X, y, lambda_n)
[n,p] = size(X);

for i=1:n 
    if y(i)>0
        d(i,1)=1;
    else
        d(i,1)=0;
    end        
end

fvec1=zeros(n*(n-1)/2,1);
fvec2=zeros(n*(n-1)/2,1);
yvec=zeros(n*(n-1)/2,1);
xmat=zeros(n*(n-1)/2,p);
for i=1:n-1 
    for j=i+1:n 
        
        a=sum((n-i+1):(n-1))+j-i;
        fvec1(a,1)=d(i,1);
        fvec2(a,1)=d(j,1);
        yvec(a,1)=y(i)-y(j);
        xmat1(a,:)=X(i,:)-X(j,:);
    end
end
xmat=xmat1(:,2:p);% xmat1(:,1)=0, so we do not need to consider this collumn
p=p-1;

beta_initial = sample_split_L1(X(:,2:p+1),y,lambda_n);

xi_Plus = yvec;
xi_Minus = zeros(n*(n-1)/2,1);
Xi1=[xi_Plus xi_Plus];
b=2/(n*(n-1));
beta1=beta_initial;
w=zeros(n*(n-1)/2,1);

QUIET    = 0;
MAX_ITER = 10000;
ABSTOL   = 1e-8;
RELTOL   = 1e-6;
rho = 1.8;
lambda_hat=(lambda_n/(rho));


for q = 1:MAX_ITER 

   %update xi_Plus and xi_Minus
   
    for i=1:n-1 
        for j=i+1:n 
            a=sum((n-i+1):(n-1))+j-i;
            xi1=-(d(i)+w(a)+rho*(xmat(a,:)*beta1-yvec(a)))/rho;
            if xi1<0
                xi1=0;
            end
            f1=d(i)*xi1+w(a)*xi1+0.5*rho*(xi1-yvec(a)+xmat(a,:)*beta1)^2;
            xi2=(w(a)+rho*(xmat(a,:)*beta1-yvec(a))-d(j))/rho;
            if xi2<0
                xi2=0;
            end
            f2=d(j)*xi2-w(a)*xi2+0.5*rho*(-xi2-yvec(a)+xmat(a,:)*beta1)^2;
            if f1<=f2
                xi2=0;
            else
                xi1=0;
            end
            xi_Plus(a)=xi1;
            xi_Minus(a)=xi2;
     
        end
    end
    %update beta
    ym=xi_Plus-xi_Minus-yvec;
    ym_hat=-(ym+w./rho);
    beta1old=beta1;
    
    beta1=glmnetFn(xmat,ym_hat,lambda_hat);
   
    %[beta1, history1] = lasso_lsqr(xmat, ym_hat, lambda_hat, 1, alpha2);
    %update w
    w=w+rho*(ym+xmat*beta1);
    
    history2.objval(q)  = (fvec1'*xi_Plus+fvec2'*xi_Minus) + lambda_n*sum(abs(beta1))/b;

    history2.tobit_iters(q) = q;
    history2.r_norm(q)  = norm(ym+xmat*beta1);
    A=[eye(n*(n-1)/2,1) -eye(n*(n-1)/2,1)];
    B=xmat;
    history2.s_norm(q)  = norm(-rho*A'*B*(beta1- beta1old));

    history2.eps_pri(q) = sqrt(n*(n-1)/2)*ABSTOL + RELTOL*max([norm([xi_Plus;xi_Minus]), norm(xmat*beta1)]);
    history2.eps_dual(q)= sqrt(n*(n-1)/2)*ABSTOL + RELTOL*norm(rho*w);

    if (history2.r_norm(q) < history2.eps_pri(q) && ...
      history2.s_norm(q) < history2.eps_dual(q))
        break;
    end

end

end
