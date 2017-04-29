%% TITLE ****************************************************************
% *                                                                      *
% *              		 521289S Machine Learning 					     *
% *                     Programming Assignment 2016                      *
% *                                                                      *
% *   Author 1: << Victoria Peredo Robinson 2515551 >>             *
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
function classify1()
%%
% This function is a place-holder for your testing code. This is not used
% by the server when validating your classifier.
%
% Since the name of this function is the same as the file-name, this
% function is called when executing classify command in the MATLAB
% given that the m-file is in the path or the current folder of MATLAB.
%
% In essence, you can use this function as a main function of your code.
% You can use this method to test code as you are implementing or
% debugging required functionalities to this file.
%
% You must not change the name of this function or the name of this file!
% However, you may add and modify the input and output variables as you
% wish to suit your needs.
%
% Typically, you could:
% - Load the data.
% - Split the data to training and validation sets.
% - Train the classifier on the training set (call trainClassifier).
% - Test it using the validation data set and learned parameters (call
%   evaluateClassifier).
% - Calculate performance statistics (accuracy, sensitivity, specificity,
%   etc.)
%
% Based on the above procedure, you can try different approaches to find
% out what would be the best way to implement the classifier training
% and evaluation.
%
% You are free to remove these comments.
%
% NOTE: FILE SYSTEM COMMANDS AND/OR SYSTEM COMMANDS ARE PROHIBITED
%       ON SERVER! PLEASE REMOVE ANY SUCH COMMANDS BEFORE SUBMISSION!
%       YOU CAN E.G. DELETE/EMPTY THIS FUNCTION AS IT IS NOT USED
%       FOR TESTING ON SERVER SIDE.

% Example: Testing a private interface subfunction:
    %myDistanceFunction( rand(10,1) , rand(10,1) )

load ('project_data.mat')

name = getNickName();

parameters = trainClassifier(trainingData,class_trainingData);

validation_result = evaluateClassifier(trainingData, parameters);

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
    nick = 'VPR';
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

X_stand = standardize(samples);
num_features = size(samples,2);


[training_data, validation_data, training_class, validation_class] = split_data(samples,classes);


fvector = zeros(num_features,1);
best_result = 0;
for in = 1:num_features
    [best_result_add, best_feature_add] = forwardsearch(training_data, training_class, fvector);
     % Update the feature vector
    fvector(best_feature_add) = 1;

    % Save best result
    if(best_result < best_result_add)
        best_result = best_result_add;
        best_fvector = fvector;
    end

end

%best_result
%parameters = struct(validation_data, training_data, best_fset, training_class);
field1 = 'f1';  value1 = best_fvector;
field2 = 'f2';  value2 = validation_data;
field3 = 'f3';  value3 = training_data;
field4 = 'f4';  value4 = training_class;
field5 = 'f5';  value5 = validation_class;

parameters = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5)

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

best_fset = parameters.f1;
validation_data = parameters.f2;
training_data = parameters.f3;
training_class = parameters.f4;
validation_class = parameters.f5;


valid_res = knnclass(validation_data, training_data, best_fset, training_class);
correct = sum(valid_res == validation_class); % amount of correct samples
validation_result = correct/length(validation_class)

results = valid_res;

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


%%
function [feat_out] =standardize(feat_in)
    N = length(feat_in);
    feat_cent = feat_in-repmat(mean(feat_in), N, 1);
    [A,B] = eig(cov(feat_cent));
    Y = sqrt(inv(B)) * A' * feat_cent';
    feat_whit = Y';
    feat_out = feat_whit;
end
%%
function[training_data, validation_data, training_class, validation_class] = split_data(data,classes)
    N = size(data,1);
    selection = randperm(N);
    training_data = data(selection(1:floor(2*N/3)), :);
    validation_data = data(selection((floor(2*N/3)+1):N), :);
    training_class = classes(selection(1:floor(2*N/3)), 1);
    validation_class = classes(selection((floor(2*N/3)+1):N), 1);
end

%%
function [best, feature] = forwardsearch(data, data_c, fvector)
    num_samples = length(data);
    best = 0;
    feature = 0;
        for in = 1:length(fvector)
            if (fvector(in) == 0)
                fvector(in) = 1;
                % Classify using k-NN
            predictedLabels = knnclass(data, data, fvector, data_c);
                correct = sum(predictedLabels == data_c); % the number of correct predictions
                result = correct/num_samples; % accuracy
                if(result > best)
                    best = result;
                    feature = in;
                end
                fvector(in) = 0;
            end
        end
end

%%
function [predictedLabels] = knnclass(dat1, dat2, fvec, classes)
    k=3;
    p1 = pdist2( dat1(:,logical(fvec)), dat2(:,logical(fvec)) );
    [D, I] = sort(p1', 1);
    I = I(1:k+1, :);
    labels = classes( : )';
    if k == 1 % this is for k-NN, k = 1
        predictedLabels = labels( I(2, : ) )';
    else % this is for k-NN, other odd k larger than 1
        predictedLabels = mode( labels( I( 1+(1:k), : ) ), 1)'; % see help mode
    end
end
