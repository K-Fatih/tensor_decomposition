clc

A=rand(1000,100);
x=rand(100,1);
y=rand(1,1000);

xref = norm(A*x);
yref = norm(y*A);

C1=GaussianProjection(A,50);
C2=srht(A,50);
C3=CountSketch(A,50);

% xc1 = norm(C1*x);
yc1 = norm(y*C2);
yc1 / yref

norm(y*C2 - y*A) / yref

function[C]=GaussianProjection(A,s)
n=size(A,2);
S=randn(n,s)/sqrt(s);
C=A*S;
end

function[C]=srht(A,s)
n=size(A,2);
sgn=randi(2,[1,n])*2-3;%onehalfare+1andtherestare−1
A=bsxfun(@times,A,sgn);%flipthesignsofeachcolumnw.p.50%
n=2^(ceil(log2(n)));
C=(fwht(A',n))';%fastWalsh−Hadarmardtransform
idx=sort(randsample(n,s));
C=C(:,idx);%subsampling
C=C*(n/sqrt(s));
end

function[C]=CountSketch(A,s)%thestreamingfashion
[m,n]=size(A);
sgn=randi(2,[1,n])*2-3;%onehalfare+1andtherestare−1
A=bsxfun(@times,A,sgn);%flipthesignsofeachcolumnw.p.50%
ll=randsample(s,n,true);%samplenitemsfrom[s]withreplacement
C=zeros(m,s);%initializeC
for j=1:n
    C(:,ll(j))=C(:,ll(j))+A(:,j);
end
end


