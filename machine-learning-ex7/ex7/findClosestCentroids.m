function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Compute m x k matrix D where D(i, j) = ||x_i - c_j||^2:
m = size(X,1);
Xnorms = arrayfun(@(idx) norm(X(idx,:), 2).^2, 1:m)'; % transpose
Cnorms = arrayfun(@(idx) norm(centroids(idx,:), 2).^2, 1:K); % don't transpose
Xnorms = repmat(Xnorms, 1, K);
Cnorms = repmat(Cnorms, m, 1);
D = Xnorms - 2*X*centroids' + Cnorms; 

% Answer is row-wise minimum of D.
[~, idx] = min(D, [], 2);



% =============================================================

end

