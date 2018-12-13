function [group,c] = kMeansCluster(X, k)

[mRow, mCol] = size(X);
temp = zeros(mRow,1);
p = randperm(mRow,k);
p = p.';
c = zeros(k,mCol);

for i =1:k
    c(i,:) = X(p(i),:);
end

dist = zeros(mRow, k);
   while 1
        for i=1:k
        observations(1,:) = c(i,:);
            for j=1:mRow
                observations(2,:) = X(j,:);
                dist(j,i) = pdist(observations);
            end
        end
        group = zeros(mRow,1);
        for i=1:mRow
                minimum = min(dist(i,:));
                p_group = find(dist(i,:) == minimum);
                group(i,:) = p_group(1,1);
        end
        if group==temp
            break;
        else
            temp = group;
        end
        
        for i=1:k
            f = find(group==i);
            if f
                c(i,:) = mean(X(find(group==i), :), 1);
            end
        end
   end
end
