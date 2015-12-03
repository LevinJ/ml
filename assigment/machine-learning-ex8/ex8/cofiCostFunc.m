function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% temp = X * Theta';
% temp = temp - Y;
% temp = temp .^ 2;
% J = sum(sum(R .* temp));
% J = J/2;

temp =  ((X * Theta')  - Y) .^ 2;
J = sum(sum(R .* temp))/2;
J = J + lambda/2 *(sum(sum(Theta .^ 2)) +  sum(sum(X .^ 2)));



% for movie=1:num_movies
%     for user=1:num_users
%         if R(movie,user) == 1
%             J = J + (Theta(user,:) * X(movie,:)' - Y(movie,user)) ^ 2;
%         end
%     end
% end
% J = J/2;

% for movie=1:num_movies
%      for user=1:num_users
%          if R(movie,user) == 1
%              X_grad(movie,:) = X_grad(movie,:) + (Theta(user,:) * X(movie,:)' - Y(movie,user)) * Theta(user,:);
%          end
%      end
% end
% 
% for user=1:num_users
%     for movie=1:num_movies
%         if R(movie,user) == 1
%             Theta_grad(user,:) = Theta_grad(user,:) + (Theta(user,:) * X(movie,:)' - Y(movie,user))  * X(movie,:);
%         end
%     end
% end

for movie=1:num_movies
idx = find(R(movie,:) == 1);
Thetatemp = Theta(idx,:);
Ytemp = Y(movie, idx);

X_grad(movie,:) = (X(movie,:) * Thetatemp' - Ytemp) * Thetatemp + lambda * X(movie,:);

end

for user=1:num_users
    
idx = find(R(:,user) == 1);
Xvalid = X(idx,:);
Yvalid = Y(idx, user);

Theta_grad(user,:) = (Xvalid * Theta(user,:)' - Yvalid)' * Xvalid + lambda  * Theta(user,:);
end



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
