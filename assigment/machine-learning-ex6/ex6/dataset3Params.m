function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% set up list to store C, Sigma, and corresponding error rate list
CVec = [0.01, 0.03, 0.1, 1, 3, 10, 30];
sigmaVec = [0.01, 0.03, 0.1, 1, 3, 10, 30];
errRate = zeros(length(CVec), length(sigmaVec));

% Try out each pair of C and Sigma, and get corresponding error rate on
% validation set

for i=1:length(CVec)
    for j=1:length(sigmaVec)
        C= CVec(i);
        sigma = sigmaVec(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        errRate(i,j) = mean( double (predictions ~= yval));
    end
end


% Return the C and sigma which correspons to lowset prediction error rate
% on validation set
[r,c]=find(errRate==min(min(errRate)));
if length(r) > 1
    r = r(1);
    c = c(1);
end 
C = CVec(r);
sigma = sigmaVec(c);




% =========================================================================

end
