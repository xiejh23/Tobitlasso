function betaHat = sample_split_L1(X,y,lambda_n)
[n,p] = size(X);
for i=1:n 
    if y(i)>0
        d(i,1)=1;
    else
        d(i,1)=0;
    end        
end
cvx_begin
    variable betaHat(p)
    W=0;
    for i=1:n/2 
        yvec(i,1)=y(2*i)-y(2*i-1);
        xmat(i,:)=X(2*i,:)-X(2*i-1,:);
        W=W+d(2*i)*max(yvec(i,1)-xmat(i,:)*betaHat,0)+d(2*i-1)*max(xmat(i,:)*betaHat-yvec(i,1),0)+lambda_n*sum(abs(betaHat));
    end
    minimize W
cvx_end
