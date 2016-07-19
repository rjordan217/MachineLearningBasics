function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
  positives = pval < epsilon;
  truePos = sum(yval & positives);
  falsePos = sum((positives - yval) > 0);
  falseNeg = sum((yval - positives) > 0);
  F1 = 2 * truePos / (2 * truePos + falsePos + falseNeg);

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    endif
end

end
