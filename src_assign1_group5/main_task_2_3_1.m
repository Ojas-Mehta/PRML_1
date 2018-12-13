[x1_training,x2_training,t1_training] = textread('..\data_assign1_group5\bivariate_group5\bivariateData\train100.txt', '%f %f %f');
[x1_test, x2_test, t1_test] = textread('..\data_assign1_group5\bivariate_group5\bivariateData\train100.txt', '%f %f %f');
[x1_valid, x2_valid, t1_valid] = textread('..\data_assign1_group5\bivariate_group5\bivariateData\val.txt', '%f %f %f');

M=60;
d=2;
sigma=0.05;

lambda=0.0;
syms x;
syms y;
[N, NotRequired] = size(x1_training);
[Ntest, NotRequired] = size(x1_test);
[Nvalid, NotRequired] = size(x1_valid);

train_size=N;
test_size=Ntest;
val_size=Nvalid;

X = [x1_training, x2_training];
X1=[x1_test,x2_test];
X2=[x1_valid,x2_valid];

[index, Centroid] = kMeansCluster(X,M);

Design=zeros(train_size,M);

Identity_M = eye(M);

for j= 1: M
 for n=1:train_size
    const = (X(n, :) - Centroid(j,:))*(X(n, :) - Centroid(j,:))'/(sigma*sigma);
    Design(n,j)=exp(-1*const);
 end
end

%Roughness of surface
Design_rough=zeros(M,M);

for i=1:M
 for j=1:M
     constant=(Centroid(i,:)-Centroid(j,:));
     constant1=(-1)*(constant)*(constant)';
     constant2=constant1/(2*sigma*sigma);
     Design_rough(i,j)=exp(constant2);
 end
end

coeff=inv((Design'*Design) + (lambda*Design_rough))*Design' * t1_training;
temp=0;
for i=1:M
   const=((x-Centroid(i,1))^2) + ((y-Centroid(i,2))^2);
   const1=exp(-1*(const)/(2*sigma*sigma));
   temp=temp+(coeff(i)*const1);
end


Design=zeros(test_size,M);
Identity_M = eye(M);
predicted_t1=zeros(test_size,1);

for j= 1: M
 for n=1:test_size
    const = (X1(n, :) - Centroid(j,:))*(X1(n, :) - Centroid(j,:))'/(sigma*sigma);
    Design(n,j)=exp(-1*const);
 end
end

predicted_t1 = Design*coeff;

rmsError = (predicted_t1-t1_test).^2;

erms = sqrt(sum(rmsError)/test_size);
scatter(predicted_t1, t1_test);
disp(erms);

g=0;
syms a b;
for i =1:M
    upper = ((a - Centroid(i,1))^2)+((b-Centroid(i,2))^2);
    upper = exp(-1*(upper)/2*(power(sigma,2)));
    g = g+coeff(i)*upper;
end
g;
fsurf(g);
axis([-20 20 -20 20 -200 200]);
hold on
scatter3(x1_training,x2_training,t1_training,'filled');
hold off;