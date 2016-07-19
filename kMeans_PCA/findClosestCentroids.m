function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

for i = 1:rows(X)
  minDistance = -1;
  idxMin = -1;

  for j = 1:K
    currentDist = sum((X(i,:) - centroids(j,:)) .^ 2);
    if (currentDist < minDistance || minDistance < 0)
      minDistance = currentDist;
      idxMin = j;
    endif
  end

  idx(i) = idxMin;
end

end
