function [ Y ] = oneHotLabels( y )
%ONEHOTLABELS Create a one-hot representation of a vector of labels.
%   y is a one-dimensional vector of length m containing labels 1, 2, ..., L.
%   We create an L x m matrix where every column j has a '1' at position y(j)
%   and zeroes everywhere else. 
%   Code taken from: http://stackoverflow.com/questions/8054258/matlab-octave-1-of-k-representation

m=length(y);
L = max(y);
Y = zeros(m,L);
Y(sub2ind(size(Y),1:m,y'))=1;
Y = Y'; % To make the result k x m
end

