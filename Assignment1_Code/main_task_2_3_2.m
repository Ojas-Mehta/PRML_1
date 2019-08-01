[x1,x2,x3,x4,x5,x6,x7,x8,t1,t2] = textread('..\data_assign1_group5\Energy_Efficiency_Dataset\energy-data.txt', '%f %f %f %f %f %f %f %f %f %f');
M=40;
d=8;

lambda=100;
Identity_M = eye(M);
sigma=50;

[N, NotRequired] = size(t1);

train_size=int16(fix(0.7*N));
val_size=int16(fix(0.2*N));
test_size=int16(fix(0.1*N));

x1_training=zeros(train_size,1);
x2_training=zeros(train_size,1);
x3_training=zeros(train_size,1);
x4_training=zeros(train_size,1);
x5_training=zeros(train_size,1);
x6_training=zeros(train_size,1);
x7_training=zeros(train_size,1);
x8_training=zeros(train_size,1);
t1_training=zeros(train_size,1);
t2_training=zeros(train_size,1);

x1_test=zeros(test_size,1);
x2_test=zeros(test_size,1);
x3_test=zeros(test_size,1);
x4_test=zeros(test_size,1);
x5_test=zeros(test_size,1);
x6_test=zeros(test_size,1);
x7_test=zeros(test_size,1);
x8_test=zeros(test_size,1);
t1_test=zeros(test_size,1);
t2_test=zeros(test_size,1);

x1_valid=zeros(val_size,1);
x2_valid=zeros(val_size,1);
x3_valid=zeros(val_size,1);
x4_valid=zeros(val_size,1);
x5_valid=zeros(val_size,1);
x6_valid=zeros(val_size,1);
x7_valid=zeros(val_size,1);
x8_valid=zeros(val_size,1);
t1_valid=zeros(val_size,1);
t2_valid=zeros(val_size,1);


for i=1 : train_size
 x1_training(i)=x1(i);
 x2_training(i)=x2(i);
 x3_training(i)=x3(i);
 x4_training(i)=x4(i);
 x5_training(i)=x5(i);
 x6_training(i)=x6(i);
 x7_training(i)=x7(i);
 x8_training(i)=x8(i);
 t1_training(i)=t1(i);
 t2_training(i)=t2(i);
end

for i=train_size+1:N-test_size-2
 x1_valid(i-train_size)=x1(i);
 x2_valid(i-train_size)=x2(i);
 x3_valid(i-train_size)=x3(i);
 x4_valid(i-train_size)=x4(i);
 x5_valid(i-train_size)=x5(i);
 x6_valid(i-train_size)=x6(i);
 x7_valid(i-train_size)=x7(i);
 x8_valid(i-train_size)=x8(i);
 t1_valid(i-train_size)=t1(i);
 t2_valid(i-train_size)=t2(i);
end

for i=(N-test_size+1):N-1
 x1_test(i-N+test_size)=x1(i);
 x2_test(i-N+test_size)=x2(i);
 x3_test(i-N+test_size)=x3(i);
 x4_test(i-N+test_size)=x4(i);
 x5_test(i-N+test_size)=x5(i);
 x6_test(i-N+test_size)=x6(i);
 x7_test(i-N+test_size)=x7(i);
 x8_test(i-N+test_size)=x8(i);
 t1_test(i-N+test_size)=t1(i);
 t2_test(i-N+test_size)=t2(i);
end

X=[x1_training,x2_training,x3_training,x4_training,x5_training,x6_training,x7_training,x8_training];
X1=[x1_test,x2_test,x3_test,x4_test,x5_test,x6_test,x7_test,x8_test];
X2=[x1_valid,x2_valid,x3_valid,x4_valid,x5_valid,x6_valid,x7_valid,x8_valid];

[index, Centroid] = kMeansCluster(X,M);

Design=zeros(train_size,M);

for j=1:M
 for n=1:train_size
    const = (X(n, :) - Centroid(j,:))*(X(n, :) - Centroid(j,:))'/(2*sigma*sigma);
    Design(n,j)=exp(-1*const);
 end
end

Design_rough=zeros(M,M);

for i=1:M
 for j=1:M
     constant=((Centroid(i,:)-Centroid(j,:)))*((Centroid(i,:)-Centroid(j,:)).');
     constant1=(-1)*(constant);
     constant2=constant1/(2*sigma*sigma);
     Design_rough(i,j)=exp(constant2);
 end
end

coeff1=(inv((Design.'*Design) + (lambda*Design_rough))*Design.') * t1_training;
coeff2=(inv((Design.'*Design) + (lambda*Design_rough))*Design.') * t2_training;

predicted_t1=zeros(test_size,1);
predicted_t2=zeros(test_size,1);

Design1=zeros(test_size,M);

for j=1:M
  for n= 1:test_size
    const = (X1(n, :) - Centroid(j,:))*(X1(n, :) - Centroid(j,:))'/(2*sigma*sigma);
    Design1(n,j)=exp(-1*const);
  end
end

predicted_t1=Design1*coeff1;
predicted_t2=Design1*coeff2;

err1 = t1_test - predicted_t1;
squareError1 = err1.^2;
meanSquareError1 = mean(squareError1);
rms1 = sqrt(meanSquareError1);
disp(rms1);

err2 = t2_test - predicted_t2;
squareError2 = err2.^2;
meanSquareError2 = mean(squareError2);
rms2 = sqrt(meanSquareError2);
disp(rms2);

ans1=0.5*(coeff1.')*(Design_rough)*(coeff1);
ans2=0.5*(coeff2.')*(Design_rough)*(coeff2);

%disp(ans1);
%disp(ans2);

% scatter(t1_test,predicted_t1);
% xlabel('t1(n)');
% ylabel('y1(xn,w)');
% title('N=537,M=30,lambda=0.000001,sigma=50');
% % 
scatter(t2_test,predicted_t2);
xlabel('t2(n)');
ylabel('y2(x,w)');
title('N=537,M=30,lambda=0.000001,sigma=50');
