function c = covariance(X)
[row,col] = size(X);
mean_vector = zeros(1,col);
for i=1:col
    mean_vector(:,i) = mean(X(:,i));
end
c = bsxfun(@minus,X,mean_vector);
c = (c.'*c)/(row-1);
end
