% Load the Dataset to be worked on
dataset_input = load('data_banknote_authentication.txt');
% assigns to corresponding values rows and cols
[rowNum, colNum] = size(dataset_input); % assignment statement

%disp(colNum);

data = dataset_input;
data(:,colNum) = 1;
labels = dataset_input(:,colNum);
labels(labels(:) == 0) = -1;

theta = zeros(1,colNum);
w = zeros(1,colNum);
T = 500;
lambda = .001;
i = 1;
b_loss = zeros(1,T);
h_loss = zeros(1,T);
for j=1:T
   w = (1/(lambda*j)) * theta;
   i = ceil(rand * numel(labels));
   update = labels(i) * dot(w, data(i,:));
   if update < 1
       theta = theta + labels(i) * data(i,:);
   end
   b_loss(j) = binaryLossAlgo(w,data,labels);
   h_loss(j) = hingeLossAlgo(w,data,labels);
end 

figure;
%plot([1:1:T],b_loss); % plot binary loss vector
plot([1:1:T],h_loss); % plot binary loss vector
hold on;

grid on; % display the major grid lines for the current axes
% Add aesthetics
title('Plotting of 3 Loss Functions (Binary, Hinge, Logistic)'); % title the figure/plot
xlabel('Algorithm Run Number'); % label the x-axis
ylabel('Loss'); % label the y-axis


% CODE REUSED FROM ASSIGNMENT 2. TESTED TO ALREADY WORK. 
% Calculates and returns the binary loss for the respective run
function binaryLoss = binaryLossAlgo(weight, data, labels)
    binaryLoss = 0; % declare accumulator for binary loss
    for i = 1:size(labels) % runs for each point
    % calculate the loss for each point and add to the accumulated loss
        lossCalc = dot(weight, data(i,:)) * labels(i);
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


%logistic loss is average hinge loss over all data points

% CODE REUSED FROM ASSIGNMENT 2. TESTED TO ALREADY WORK. 
% Calculates and returns the logistic loss for the respective run
function logisticLoss = hingeLossAlgo(weight, data, labels)
    logisticLoss = 0; % declare accumulator for logistic loss
    for i = 1:size(labels) % runs for each point
        lossCalc = 1 - labels(i) * dot(weight, data(i,:));
        lossCalc = max(0, lossCalc); % take the maximum of the two
        logisticLoss = logisticLoss + lossCalc; % update logistic loss
    end
    % calculate logistic loss using formula from L14 S15
    logisticLoss = logisticLoss / i; % perform calculation
end

