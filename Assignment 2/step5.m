% Load the Dataset to be worked on
dataset_input = load('irisnum.txt'); % load input data points and label output

runAlgoNum = 20; % define the number times to run perception algorithm
i = 1; % declare counter i

% Size returns the number of rows and columns of the matrix and
% assigns to corresponding values rows and cols
[rowNum, colNum] = size(dataset_input); % assignment statement

% Declare matrices/vectors used to store loss information for each run
binaryLossVector = zeros(1, runAlgoNum); % vector for binary loss data
hingeLossVector = zeros(1, runAlgoNum); % vector for hinge loss data
logisticLossVector = zeros(1, runAlgoNum); % vector for logistic loss data
weightVector = zeros(runAlgoNum, colNum); % declare a 1 x colNum+1 matrix consisting of zeros

while i <= runAlgoNum 
    % Establish our matrix of values which contains both the input and
    % corresponding output values. Matrix will have the same number of rows
    % as the input (and output) and the number of columns will be equal to
    % the input plus the output.
    matrixM = dataset_input; % set matrixM equal to both the datapoints and labels
    matrixM = matrixM(randperm(size(matrixM, 1)), :); % Randomize order of values.
    dataset_output_mod = matrixM(:,end); % Matrix for corresponding label outputs.

    % Iris Setosa vs All. Separate Class 1 from 2 & 3.
    dataset_output_mod(dataset_output_mod(:,end)==1) = 1; % separate
    dataset_output_mod(dataset_output_mod(:,end)==2) = -1; % separate
    dataset_output_mod(dataset_output_mod(:,end)==3) = -1; % separate

    % Iris Versicolour vs All. Separate Class 2 from 1 & 3.
    %dataset_output_mod(dataset_output_mod(:,end)==1) = -1; % separate
    %dataset_output_mod(dataset_output_mod(:,end)==2) = 1; % separate
    %dataset_output_mod(dataset_output_mod(:,end)==3) = -1; % separate

    % Iris Virginica vs All. Separate Class 3 from 1 & 2.
    %dataset_output_mod(dataset_output_mod(:,end)==1) = -1; % separate
    %dataset_output_mod(dataset_output_mod(:,end)==2) = -1; % separate
    %dataset_output_mod(dataset_output_mod(:,end)==3) = 1; % separate

    matrixM(:,end) = 1; % add constant one coordinate

    % Similar to what we did in step 2, but instructions imply that we send
    % the entire combined matrix as opposed to just the randomized input
    % data point matrix.
    % Call the function that returns the calculated weight vector.
    weightVector(i,:) = perceptronAlgo(matrixM, dataset_output_mod); %assignment statment    

    % Calculate and store binary loss, hinge loss, logistic loss for
    % respective run
    binaryLossVector(i) = binaryLossAlgo(weightVector(i,:), matrixM, dataset_output_mod); % calculate and assign
    hingePoint = randi(rowNum); % select random point from input data
    hingeLossVector(i) = hingeLossAlgo(weightVector(i,:), matrixM(hingePoint,:), dataset_output_mod(hingePoint,:)); % calculate and assign
    logisticLossVector(i) = logisticLossAlgo(weightVector(i,:), matrixM, dataset_output_mod); % calculate and assign
    disp(i); % display run number
    i = i + 1; % increment counter
end

disp('Binary Loss'); % display text stating information
[binaryMin, binaryIndex] = min(binaryLossVector); % get smallest binary loss
disp('Run Number of Min'); % display text stating information
disp(binaryIndex); % display index of smaller binary loss
disp('Minimum'); % display text stating information
disp(binaryMin); % display smallest binary loss
disp('Weight Vector'); % display text stating information
disp(weightVector(binaryIndex,:)); % display weight vector associated
disp('Hinge Loss'); % display text stating information
[hingeMin, hingeIndex] = min(hingeLossVector); % get smallest hinge loss
disp('Run Number of Min'); % display text stating information
disp(hingeIndex); % display index of smallest hinge loss
disp('Minimum'); % display text stating information
disp(hingeMin); % display smallest hinge loss
disp('Weight Vector'); % display text stating information
disp(weightVector(hingeIndex,:)); % display weight vector associated
disp('Logistic Loss'); % display text stating information
[logisticMin, logisticIndex] = min(logisticLossVector); % get smallest logistic loss
disp('Run Number of Min'); % display text stating information
disp(logisticIndex); % display index of smallest logistic loss
disp('Minimum'); % display text stating information
disp(logisticMin); % display smallest logistic loss
disp('Weight Vector'); % display text stating information
disp(weightVector(logisticIndex,:)); % display weight vector associated

figure; % create a new figure window with default property values

plot([1:1:runAlgoNum],binaryLossVector,'r'); % plot binary loss vector
hold on; % retains plots in the current axes so new plots do not delete it
plot([1:1:runAlgoNum],hingeLossVector,'g'); % plot hinge loss vector
hold on; % retains plots in the current axes so new plots do not delete it
plot([1:1:runAlgoNum],logisticLossVector,'b'); % plot logistic loss vector
hold on; % retains plots in the current axes so new plots do not delete it

grid on; % display the major grid lines for the current axes

% Add aesthetics
title('Plotting of 3 Loss Functions (Binary, Hinge, Logistic)'); % title the figure/plot
legend('Binary Loss','Hinge Loss','Logistic Loss'); % label the legend
xlabel('Algorithm Run Number'); % label the x-axis
ylabel('Losses'); % label the y-axis
xlim([0,20]); % set domain of possible values to be displayed on plot
xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]); % x-axis #'s
xtickangle(45); % make x-axis #'s display on a 45 degree tilt
ylim([-0.05,1]); % set range of possible values to be displayed on plot


% Function that returns the normalized weight vector.
function weightVector = perceptronAlgo(dataset_input_mod, dataset_output_mod)
    % Size returns the number of rows and columns of the matrix and
    % assigns to corresponding values rows and cols  
    [rowNumLoc,colNumLoc] = size(dataset_input_mod); % assignment statement
    weightVector = zeros(1,colNumLoc); % declare a 1x2 matrix consisting of zeros
    updateNum = 0; % declare a counter for number of updates
    marker = -1; % declare a marker indicating when more training is needed
    j = 1; % declare counter j

    % While loop. Loops until conditions are no longer satisfied.
    while j <= 100 && marker == -1
        marker = 0; % unmark marker
        k = 1; % declare counter k
        while k <= rowNumLoc % cycle through the six data points
            % Perform calculations which we'll use to determine if an
            % update is warranted and if more training is required.
            signValue = dataset_output_mod(k) * dot(weightVector, dataset_input_mod(k,:));
            if signValue <= 0 % if negative or equal to 0
                % update weight vector
                weightVector = weightVector + dataset_output_mod(k) * dataset_input_mod(k,:);
                updateNum = updateNum + 1; % increment update counter
                marker = -1; % remark marker
            end
            k = k + 1; % increment counter
        end
        j = j + 1; % increment counter  
    end
    % Perform an array right division using the weight vector and its
    % 2-norm or maximum singular value of the matrix.
    weightVector = weightVector ./ norm(weightVector); % perform calculation
end

% Calculates and returns the hinge loss for the respective run
function hingeLoss = hingeLossAlgo(weightVector, dataset_input_mod, dataset_output_mod)
    % calculate hinge loss using formula
    % from L14 S37: hinge loss = max{0, 1 - t<w,x>}
    hingeLoss = max(0,1 - dataset_output_mod * dot(weightVector,dataset_input_mod));
end

% Calculates and returns the binary loss for the respective run
function binaryLoss = binaryLossAlgo(weightVector, dataset_input_mod, dataset_output_mod)
	binaryLoss = 0; % declare accumulator for binary loss
    for i = 1:size(dataset_output_mod) % runs for each point
        % calculate the loss for each point and add to the accumulated loss
	    lossCalc = dot(weightVector, dataset_input_mod(i,:)) * dataset_output_mod(i);
	    lossCalc = lossCalc / abs(lossCalc); % perform calculation
	    lossCalc = (.5 * lossCalc - .5); % perform calculation
	    if lossCalc ~= 0 % if signValue not equal to 0
	        lossCalc = 1; % set equal to 1 if not equal to 0
	    end
	    binaryLoss = binaryLoss + lossCalc; % update binary loss
    end
    % calculate binary loss using formula from L14 S32
	binaryLoss = binaryLoss / i; % perform calculation
end

% Calculates and returns the logistic loss for the respective run
function logisticLoss = logisticLossAlgo(weightVector, dataset_input_mod, dataset_output_mod)
    logisticLoss = 0; % declare accumulator for logistic loss
    for i = 1:size(dataset_output_mod) % runs for each point
        lossCalc = 1 - dataset_output_mod(i) * dot(weightVector, dataset_input_mod(i,:));
        lossCalc = max(0, lossCalc); % take the maximum of the two
        logisticLoss = logisticLoss + lossCalc; % update logistic loss
    end
    % calculate logistic loss using formula from L14 S15
    logisticLoss = logisticLoss / i; % perform calculation
end