function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


for feature=1:n
    temp_mu_sum = 0;
    for example=1:m
        temp_mu_sum = temp_mu_sum + X(example,feature);
    end
    mu(feature) = temp_mu_sum/m;
end

for feature=1:n
    temp_sigma_sum = 0;
    for example=1:m
        temp_sigma_sum = temp_sigma_sum + (X(example,feature)-mu(feature)) ^2;
    end
    sigma2(feature) = temp_sigma_sum/m;
end







% =============================================================


end
