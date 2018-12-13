[X1,Y1] = textread('..\data_assign1_group5\univariate_data.txt', '%f %f');

M=6;
N=20;

X=X1(1:N);
Y=Y1(1:N);

coeff=ones(M,1);

for i=1:N
    Noise=normrnd(0,0.1);
    Y(i)=exp(cos(2*pi*X(i)))+X(i)+Noise;
end

lambda=0.0; %regularization constant
Design = zeros(N, M+1);
monomial = @(x, n) x.^n;
Identity_M = eye(M+1);

for j = 1:M+1
    for i = 1:N
        Design(i,j) = monomial( X(i), j-1);
    end
end
%disp(size(Design));

coeff = inv((Design')*Design + (lambda*Identity_M)) * Design' * Y;
%coeff = pinv(Design)*Y;

Ntest = 20;
predicted=zeros(Ntest,1);
%disp(size(coeff));

Xtest = rand(Ntest,1);
Ytest = rand(Ntest,1);

for i= 1:Ntest
    Noise=normrnd(0,0.1);
    Ytest(i) = exp(cos(2*pi*Xtest(i))) + Xtest(i)+Noise;
end

err_rms_vect = zeros(Ntest, 1);
for n=1:20
    val=0;
    for j=1:M+1
      val=val+coeff(j)*monomial(X(n),j-1);
    end
    predicted(n)=val+0.2;
    err_rms_vect(n) = (Y(n) - predicted(n)).^2;
end

err_rms = sqrt(sum(err_rms_vect)/Ntest);
B = fliplr(coeff.');
p=poly2sym(B);

s=[-5.0:0.01:5.0];
mm= exp(cos(2*pi*s))+s;

%plot (s,mm,'g'),xlabel('x'),ylabel('t'),title('M=8');
x=[-5,-4,-3,-2,-1,0,1,2,3,4,5];

%fplot( p, 'r');
% h=zeros(3,1);

%fplot(mm, 'g');
%  hold on
%  axis([0 1 -10 10]);
%  h(1)=fplot(p, 'r');
%  h(2)=plot(s, mm, 'g');
%  h(3)=scatter(X, Y,'m');
%  hold off;
%  
%  legend(h,'y(x,w)','f(x)','Training Data');
%  title('M=6');
scatter(Y,predicted);
disp(err_rms);