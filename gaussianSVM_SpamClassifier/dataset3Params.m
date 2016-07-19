function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%


Cs = [.01 .03 .1 .3 1 3 10 30];
sigmas = [.01 .03 .1 .3 1 3 10 30];


minErrRate = 1;

for i = 1:8
  for j = 1:8
    model = svmTrain(X, y, Cs(i), @(x1, x2) gaussianKernel(x1, x2, sigmas(j)));
    predicted = svmPredict(model, Xval);
    meanError = mean(double(predicted ~= yval));
    if (meanError < minErrRate)
      minErrRate = meanError;
      idxC = i;
      idxSigma = j;
    endif
  end
end

C = Cs(idxC);
sigma = sigmas(idxSigma);

end
