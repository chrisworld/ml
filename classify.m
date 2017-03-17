%% TITLE ****************************************************************
% *                                                                      *
% *              		 521289S Machine Learning 					     *
% *                     Programming Assignment 2016                      *
% *                                                                      *
% *   Author 1: Christian Walter 2516301                                 *
% *                                                                      *
% *   NOTE: The file name for this file MUST BE 'classify.m'!            *
% *         Everything should be included in this single file.           *
% *                                                                      *
% ************************************************************************

%% NOTE ******************************************************************
% *                                                                      *
% *       DO NOT DEFINE ANY GLOBAL VARIABLES (outside functions)!        *
% *                                                                      *
% *       Your task is to complete the PUBLIC INTERFACE below without    *
% *       modifying the definitions. You can define and implement any    *
% *       functions you like in the PRIVATE INTERFACE at the end of      *
% *       this file as you wish.                                         *
% *                                                                      *
% ************************************************************************

%% HINT ******************************************************************
% *                                                                      *
% *       If you enable cell folding for the m-file editor, you can      *
% *       easily hide these comments from view for less clutter!         *
% *                                                                      *
% *       [ File -> Preferences -> Code folding -> Cells = enable.       *
% *         Then use -/+ signs on the left margin for hiding/showing. ]  *
% *                                                                      *
% ************************************************************************

%% This is the main function of this m-file
%  You can use this e.g. for unit testing.
%
% INPUT:  none (change if you wish)
%
% OUTPUT: none (change if you wish)
%%

function classify()
%% - Load the data
close all
clear all
clc
load training_data

%% - Split the data to training and validation sets.
N = size(trainingData,1);
N = 1000;    %-%
selection = randperm(N);
training_data = trainingData(selection(1:floor(2*N/3)), :);
training_class = class_trainingData(selection(1:floor(2*N/3)), :);
validation_data = trainingData(selection((floor(2*N/3)+1):N), :);
validation_class = class_trainingData(selection((floor(2*N/3)+1):N), :);

%plotFeatureSamples(training_data, training_class)

%% - Train the classifier on the training set (call trainClassifier).
parameters = trainClassifier( training_data, training_class );

 
%% - Test it using the validation data set and learned parameters (call
%   evaluateClassifier).
results = evaluateClassifier( validation_data, parameters );

%% - Calculate performance statistics (accuracy, sensitivity, specificity,
%   etc.)

correct = sum(results == validation_class);
correct_classified = correct / length(validation_class)

end


%% PUBLIC INTERFACE ******************************************************
% *                                                                      *
% *   Below are the functions that define the classifier training and    *
% *   evaluation. Your task is to complete these!                        *
% *                                                                      *
% *   NOTE: You MUST NOT change the function definitions that describe   *
% *         the input and output variables, and the names of the         *
% *         functions! Otherwise, the automatic ranking system cannot    *
% *         evaluate your algorithm!                                     *
% *                                                                      *
% ************************************************************************


%% This function gives the nick name that is shown in the ranking list
% at the course web page. Use 1-15 characters (a-z, A-Z, 0-9 or _).
%
% Check the rankings page beforehand to guarantee an unique nickname:
% http://www.ee.oulu.fi/research/tklab/courses/521289S/progex/rankings.html
% 
% INPUT:  none
%
% OUTPUT: Please change this to be a unique name and do not alter it 
% if resubmitting a new version to the ranking system for re-evaluation!
%%
function nick = getNickName()
    nick = 'lazy_Flegmon';   % CHANGE THIS!
end


%% This is the training interface for the classifier you are constructing.
%  All the learning takes place here.
%
% INPUT:  
%
%   samples:
%            A N-by-M data matrix. The rows represent samples and 
%            the columns features. N is the number of samples.
%            M is the number of features.
%
%            This could be e.g. the training data matrix given on the 
%            course web page or the validation data set that has been 
%            withheld for the validation on the server side.
%
%            Note: The value for N can vary! Do not hard-code it!
%
%   classes:
%
%            A N-by-1 vector of correct classes. Each row gives the correct
%            class for the corresponding data in the samples matrix.
%
% OUTPUT: 
%
%   parameters:
%            Any type of data structure supported by MATLAB. You decide!
%            You should use this to store the results of the training.
%
%            This set of parameters is given to the classifying function
%            that can operate on a completely different set of data.
%
%            For example, a classifier based on discriminant functions
%            could store here the weight vectors/matrices that define 
%            the functions. A kNN-classifier would store all the training  
%            data samples, their classification, and the value chosen 
%            for the k-parameter.
%            
%            Especially, structure arrays (keyword: struct) are useful for 
%            storing multiple parameters of different type in a single 
%            struct. Cell arrays could also be useful.
%            See MATLAB help for details on these.
%%
function parameters = trainClassifier( samples, classes )
%% Comments
% Insert the function body here!
%
% You must be able to construct a classifier solely based on the data in
% the parameters data structure in the evaluateClassifier-function.
% Consequently, choose carefully what is needed to be stored in the
% parameters data structure.
%
% Hint 1: If you wish to avoid overtraining, you could partition the
% training samples to a actual training data set and a testing data set
% that is used for a stopping/overtraining criterion.
%
% Hint 2: To avoid duplicating e.g. classifier construction code define
% suitable own functions at the end of this file as subfunctions!
% You could also utilize nested functions within this function
% to perform repetitive tasks. See MATLAB help for details.
%
% You are free to remove these comments.
%

%% Train feature vector
k = 1;
num_features = size(samples,2);
fvector = zeros(num_features,1);
best_result = 0;
for in = 1:num_features
    [best_result_add, best_feature_add] = ...
        forwardsearch(samples, classes, fvector, k);
    % Update the feature vector  
    fvector(best_feature_add) = 1;
  
    % Save best result
    if(best_result < best_result_add)
        best_result = best_result_add;
        best_fvector = fvector;
    end
end

parameters = struct('training_samples', samples, ...
    'training_class', classes, 'best_fvector', best_fvector, 'k', k);

end


%% This is the evaluation interface of your classifier.
%  This function is used to perform the actual classification of a set of
%  samples given a fixed set of parameters defining the classifier.
%
% INPUT:   
%   samples:
%            A N-by-M data matrix. The rows represent samples and 
%            the columns features. N is the number of samples.
%            M is the number of features.
%
%            Note that N could be different from what it was in the
%            previous training function!
%
%   parameters:
%            Any type of data structure supported by MATLAB.
%
%            This is the output of the trainClassifier function you have
%            implemented above.
%
% OUTPUT: 
%   results:
%            The results of the classification as a N-by-1 vector of
%            estimated classes.
%
%            The data type and value range must correspond to the classes
%            vector in the previous function.
%%
function results = evaluateClassifier( samples, parameters )
%% Comments
% Insert the function body here!
%
% Typically, you must construct the classifier with the given parameters
% and then apply it to the given data.
%
% Note that no learning should take place in this function! The classifier
% is just applied to the data.
%
% Hint: To avoid duplicating e.g. classifier construction code define
% suitable own functions at the end of this file as subfunctions!
% You could also utilize nested functions within this function
% to perform repetitive tasks. See MATLAB help for details.
%
% You are free to remove these comments.
%

%% Test results. Train the system and evaluate the accuracy.
results = knnclass(samples, parameters.training_samples, ...
    parameters.best_fvector, parameters.training_class, parameters.k);

end


%% PRIVATE INTERFACE *****************************************************
% *                                                                      *
% *   User defined functions that are needed e.g. for training and       *
% *   evaluating the classifier above.                                   *
% *                                                                      *
% *   Please note that these are subfunctions that are visible only to   *
% *   the other functions in this file. These are defined using the      *
% *   'function' keyword after the body of the preceding functions or    *
% *   subfunctions. Subfunctions are not visible outside the file where  *
% *   they are defined.                                                  *
% *                                                                      *
% *   To avoid calling MATLAB toolbox functions that are not available   *
% *   on the server side, implement those here.                          *
% *                                                                      *
% ************************************************************************
%
%% KNN classification
function [predictedLabels] = knnclass(dat1, dat2, fvec, classes, k)

    p1 = pdist2( dat1(:,logical(fvec)), dat2(:,logical(fvec)) );
    % Here we aim in finding k-smallest elements
    [D, I] = sort(p1', 1);

    I = I(1:k+1, :);
    labels = classes( : )';
    % this is for k-NN, k = 1
    if k == 1 
        predictedLabels = labels( I(2, : ) )';
    % this is for k-NN, other odd k larger than 1
    else 
        predictedLabels = mode( labels( I( 1+(1:k), : ) ), 1)';
    end
end

%% Forwardsearch
function [best, feature] = forwardsearch(data, data_c, fvector, k)
    % SFS, from previous lesson.
    num_samples = length(data);
    best = 0;
    feature = 0;
    
    for in = 1:length(fvector)
        if (fvector(in) == 0)
            fvector(in) = 1;
            % Classify using k-NN
	        predictedLabels = knnclass(data, data, fvector, data_c, k);
            % the number of correct predictions
            correct = sum(predictedLabels == data_c); 
            result = correct/num_samples; % accuracy
            if(result > best)
                best = result; 
                feature = in; 
            end
            fvector(in) = 0;
        end
    end
end

%% plotFeatureSamples
function plotFeatureSamples(samples, class)
samples(:, 2)
class = class == 1
samples = samples(:, 1) .* class
samples(samples==0) = nan

N = length(samples)
% steps
stepsize = 25;
x = min(samples):stepsize:max(samples);

% 1.histogramm
h1 = histcounts(samples, x);
h1 = h1/sum(h1)/stepsize;

% plot 1.histogramm
bincenters = x(1:end-1) + diff(x)/2;
figure
plot(bincenters, h1)
hold on
end




%% EOF
