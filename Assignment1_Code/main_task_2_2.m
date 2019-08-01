[x1, x2, t] = textread('..\data_assign1_group5\bivariate_group5\bivariateData\train20.txt', '%f %f %f');
[x1test, x2test, ttest] = textread('..\data_assign1_group5\bivariate_group5\bivariateData\train20.txt', '%f %f %f');
[Ntest, NotRequired] = size(ttest);
syms x y;
d = 2; %Number of variables
    M = 5; %Degree of polynomial
    lambda=2; %regularization constant
    [N, ~] = size(t); %number of examples
    D = factorial(M+d) / (factorial(d)*factorial(M)); 
    expression = expand((x+y).^0);  
    for i = 1:M
        expression = expression + expand((x+y).^i);
    end
    [notrequired, terms] = coeffs(expression, [x,y]);   

    Design = zeros(N, D);
    Identity_M = eye(D);
    for j = 1:D
        for i = 1:N
            currentTerm = terms(j);
            currentTerm = subs(currentTerm, x, x1(i));
            currentTerm = subs(currentTerm, y, x2(i));
            Design(i,j) = currentTerm;
            
        end
    end

    coeff = ((Design')*Design + (lambda*Identity_M)) \ (Design' * t); 
    modelFunction = coeff.* terms.';
    predicted_output = zeros(Ntest, 1);
    err_rms_vect = zeros(Ntest, 1);
    for i = 1:Ntest
        model_output = 0;
        for j = 1:D
            currentTerm = modelFunction(j);
            currentTerm = subs(currentTerm, x, x1test(i));
            currentTerm = subs(currentTerm, y, x2test(i));
            model_output = model_output + currentTerm;
        end
        predicted_output(i) = model_output;
        err_rms_vect(i) = (ttest(i) - model_output).^2;
    end
    
ans1=terms*coeff;
fs = fsurf(ans1,[-100 100],'-','EdgeColor','none');
%  cc = jet(5);
%  colormap(cc);
 mycolors = [0 0.5 1];
colormap(mycolors);
% direction = [0 0 1];
% rotate(fs,direction,90);
 
 fs.FaceAlpha = 0.4;
%  fs.FaceColor: 'interp';

hold on

scatter3(x1test,x2test,t,60,'red', 'filled');
axis([-30 30 -30 30 -60 60]);
xlabel('x1');
ylabel('x2');
zlabel('y(x1,x2,w)');
hold off
title('Plot of x1,x2 and y(x1,x2,w)');
    err_rms = sqrt(sum(err_rms_vect)/Ntest);
    disp(err_rms);
legend('Predicted Model','Actual output(t_t_r_a_i_n)');

