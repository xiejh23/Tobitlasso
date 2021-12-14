function ind = selvarFn(betaHat)
p=size(betaHat);
t=1;

for i=1:p 
    if betaHat(i)>0.00001||betaHat(i)<-0.00001
        ind(t)=i+1;
        t=t+1;
        
    end
    
end

if max(abs(betaHat))<0.00001
    ind=0;
    
end
end
