clear;
clc;
p=100;
n=100;
true_ind=[2 3 4 5 6];
beta=[0 1 1/2 1/3 1/4 1/5  zeros(1,p-6)]';
gama=0.1;
M=100;
c=1.1;
alpha=0.05;
lambda_n = 0.4*c*sqrt((log(2*p/alpha))/(n-1));
for m=1:M 
%%
%data generating process
    [X,y]=cauchy_data_gp(p,n,beta,gama);

%%
% estimated by linear lasso
    betaHat1 = glmnetFn(X,y,lambda_n);
    beta1{1,m}=betaHat1;
%%
%estimated by l1-PD-LAD with LASSO initial
    [betaHat2, history2]=tobitlasso_admm_Fun(X, y, lambda_n);
    beta2{1,m}=betaHat2;
%%
%estimated by l1-PD-LAD with split-sample estimator (l1-SS) initial
    betaHat3=tobitlasso_admm_Fun_L1_ss(X, y, lambda_n);
    beta3{1,m}=betaHat3;
%%
%selection of variables by l1-PD-LAD with LASSO initial
     ind = selvarFn(betaHat2);
     index1{1,m}=ind';

%%
% post-l1-PD-LAD with OLS initial
     Xs=X(:,ind);
     betaHat4 = tobitADMM_Fun(Xs, y);
     betapost1{1,m}=betaHat4;
%%
%selection of variables by l1-PD-LAD with split-sample estimator (l1-SS) initial
     ind2 = selvarFn(betaHat3);
     index2{1,m}=ind2';

%%
% post-l1-PD-LAD with split-sample estimator (l1-SS) initial
     Xs=X(:,ind2);
     betaHat5 = tobitADMM_Fun_ss(Xs, y);
     betapost2{1,m}=betaHat5;
         
%%
%regession by Oracle
     Xt=X(:,true_ind);
     betaHat6 = tobitADMM_Fun(Xt,y);
          %betaLP3 = tobitFn(Xt,y);
     betaoracle1{1,m}=betaHat6;
%%
% estimated by l1-penalized split-sample estimator (l1-SS)
     betaHat7 = sample_split_L1(X,y,lambda_n);
     beta7{1,m}=betaHat7;

end

%%
% performance results
for m=1:M
    %%
    %for mean L0
    beta1_L0(m)=sum(size(find(abs(beta1{1,m})>0.00001),1));
    beta2_L0(m)=sum(size(find(abs(beta2{1,m})>0.00001),1));
    beta3_L0(m)=sum(size(find(abs(beta3{1,m})>0.00001),1));
    beta4_L0(m)=sum(size(find(abs(betapost1{1,m})>0.00001),1));
    beta5_L0(m)=sum(size(find(abs(betapost2{1,m})>0.00001),1));
    beta6_L0(m)=sum(size(find(abs(betaoracle1{1,m})>0.00001),1));
    beta7_L0(m)=sum(size(find(abs(beta7{1,m})>0.00001),1));
    
    %%
    %for mean L0_S
    bh1=beta1{1,m};
    bh2=[0;beta2{1,m}];
    bh3=[0;beta3{1,m}];
    bh4=zeros(p,1);
    bh4(index1{1,m})=betapost1{1,m};
    bh5=zeros(p,1);
    bh5(index2{1,m})=betapost2{1,m};
    bh6=zeros(p,1);
    bh6(true_ind)=betaoracle1{1,m};
    bh7=beta7{1,m};
    beta1_L0_S(m)=sum(size(find(abs(bh1(2:6))>0.00001),1));
    beta2_L0_S(m)=sum(size(find(abs(bh2(2:6))>0.00001),1));
    beta3_L0_S(m)=sum(size(find(abs(bh3(2:6))>0.00001),1));
    beta4_L0_S(m)=sum(size(find(abs(bh4(2:6))>0.00001),1));
    beta5_L0_S(m)=sum(size(find(abs(bh5(2:6))>0.00001),1));
    beta6_L0_S(m)=sum(size(find(abs(bh6(2:6))>0.00001),1));
    beta7_L0_S(m)=sum(size(find(abs(bh7(2:6))>0.00001),1));
    
    %%
    %for mean L1_S
    beta1_L1_S(m)=sum(abs(bh1(2:6)));
    beta2_L1_S(m)=sum(abs(bh2(2:6)));
    beta3_L1_S(m)=sum(abs(bh3(2:6)));
    beta4_L1_S(m)=sum(abs(bh4(2:6)));
    beta5_L1_S(m)=sum(abs(bh5(2:6)));
    beta6_L1_S(m)=sum(abs(bh6(2:6)));
    beta7_L1_S(m)=sum(abs(bh7(2:6)));
    
    %%
    %for rmse
    beta1_r(m)=sum((bh1-beta).^2);
    beta2_r(m)=sum((bh2-beta).^2);
    beta3_r(m)=sum((bh3-beta).^2);
    beta4_r(m)=sum((bh4-beta).^2);
    beta5_r(m)=sum((bh5-beta).^2);
    beta6_r(m)=sum((bh6-beta).^2);
    beta7_r(m)=sum((bh7-beta).^2);
    
   %%
    %for bias
    beta1_b(:,m)=bh1-beta;
    beta2_b(:,m)=bh2-beta;
    beta3_b(:,m)=bh3-beta;
    beta4_b(:,m)=bh4-beta;
    beta5_b(:,m)=bh5-beta;
    beta6_b(:,m)=bh6-beta;
    beta7_b(:,m)=bh7-beta;
end

vec_L0=[sum(beta1_L0);sum(beta2_L0);sum(beta4_L0);sum(beta7_L0);sum(beta3_L0);sum(beta5_L0);sum(beta6_L0)]/M;
vec_L0_S=[sum(beta1_L0_S);sum(beta2_L0_S);sum(beta4_L0_S);sum(beta7_L0_S);sum(beta3_L0_S);sum(beta5_L0_S);sum(beta6_L0_S)]/M;
vec_L1_S=[sum(beta1_L1_S);sum(beta2_L1_S);sum(beta4_L1_S);sum(beta7_L1_S);sum(beta3_L1_S);sum(beta5_L1_S);sum(beta6_L1_S)]/M;
vec_rmse=sqrt([sum(beta1_r);sum(beta2_r);sum(beta4_r);sum(beta7_r);sum(beta3_r);sum(beta5_r);sum(beta6_r)]/M);
vec_bias=[norm(mean(beta1_b,2));norm(mean(beta2_b,2));norm(mean(beta4_b,2));norm(mean(beta7_b,2));norm(mean(beta3_b,2));norm(mean(beta5_b,2));norm(mean(beta6_b,2))];

Table=[vec_L0 vec_L0_S vec_L1_S vec_bias vec_rmse];
