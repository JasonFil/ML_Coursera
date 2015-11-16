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
worstMeanError = 1; % On average, make an error every time.
Cvals = [0.1, 0.3, 1, 3, 9, 27, 81];
svals = [0.01, 0.03, 0.1, 0.3, 1];
for i = 1:length(Cvals)
    c_curr = Cvals(i);
    for j = 1:length(svals)
        s_curr = svals(j);
        fprintf('\tTraining model for C=%.1f and s = %.2f...\n', c_curr, s_curr);
        model= svmTrain(X, y, c_curr, @(x1, x2) gaussianKernel(x1, x2, s_curr)); % Not using two last arguments.
        fprintf('\tEvaluating model for C=%.1f and s = %.2f...\n', c_curr, s_curr);
        predictions = svmPredict(model, Xval);
        meanerr = mean(double(predictions ~= yval)); % mean error. 
        if(meanerr < worstMeanError)
            worstMeanError = meanerr;
            C = c_curr;
            sigma = s_curr;
        end
    end
end

fprintf('Best C=%.1f and best s=%.2f.\n', C, sigma)



% =========================================================================

end
