% Load the Dataset to be worked on
dataset_input = load('6pointsinputs.txt'); % load input data points
dataset_output = load('6pointsoutputs.txt'); % load corresponding binary outputs

dataset_input = dataset_input * 10; % scale data points by a factor of 10

runAlgoNum = 20;

[rowNum, colNum] = size(dataset_input);
matrixM = zeros(rowNum, colNum+1);

weightVector = zeros(runAlgoNum, colNum+1);
binaryLossVector = zeros(1, runAlgoNum);
hingeLossVector = zeros(1, runAlgoNum);
logisticLossVector = zeros(1, runAlgoNum);

i = 1;

while i <= runAlgoNum
   
    matrixM(:,1:colNum) = dataset_input;
    matrixM(:,end) = dataset_output;
    matrixM = matrixM(randperm(size(matrixM, 1)), :);


    dataset_input_mod = matrixM(:,1:colNum);
    dataset_output_mod = matrixM(:,end);
    matrixM(:,end) = 1;
    
    weightVector(i,:) = perceptronAlgo(matrixM, dataset_output_mod);
    binaryLossVector(i) = binaryLossAlgo(weightVector(i,:), matrixM, dataset_output_mod);

    
    %randomly select a point from data_matrix and data_labels to find 
    %hinge error for
    hpoint = randi(rowNum);
    hingeLossVector(i) = hingeLossAlgo(weightVector(i,:), matrixM(hpoint,:), dataset_output_mod(hpoint,:));
    logisticLossVector(i) = logisticLossAlgo(weightVector(i,:), matrixM, dataset_output_mod);

    disp(i);
    i = i + 1;
end

figure;
plot([1:1:runAlgoNum],binaryLossVector,'r');
hold on;
plot([1:1:runAlgoNum],hingeLossVector,'g');
hold on;
plot([1:1:runAlgoNum],logisticLossVector,'b');
hold on;

grid on;

title('Plotting of 3 Loss Functions (Binary, Hinge, Logistic)');
legend('Binary Loss','Hinge Loss','Logistic Loss');
xlabel('Algorithm Run Number');
ylabel('Losses');
xlim([0,20]);
xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]);
xtickangle(45);
ylim([-0.05,1]);



function weightVector = perceptronAlgo(dataset_input_mod, dataset_output_mod)
    [rowNumLoc,colNumLoc] = size(dataset_input_mod);
    updateNum = 0;
    marker = -1;
    weightVector = zeros(1,colNumLoc);
    j = 1; % declare counter j

    % While loop. Loops until conditions are no longer satisfied.
    while j <= 100 && marker == -1
        marker = 0; % unmark marker
        k = 1; % declare counter k
        while k <= rowNumLoc % cycle through the six data points
            signValue = dataset_output_mod(k) * dot(weightVector, dataset_input_mod(k,:));
            if signValue <= 0
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


function hingeLoss = hingeLossAlgo(weight, data, labels)
    % hinge loss = max(0, 1 - t<w,x>)
    hingeLoss = max(0,1 - labels * dot(weight,data));
end

function binaryLoss = binaryLossAlgo(weight, data, labels)
    % compute average binary loss for given weight vector over all 
    % inputs and respective labels
	binaryLoss = 0;
    for i=1:size(labels)
	    sign = dot(weight,data(i,:)) * labels(i);
	    sign = sign / abs(sign);
	    sign = (.5 * sign - .5);
	    if sign ~= 0
	        sign = 1;
	    end
	    binaryLoss = binaryLoss + sign;
    end
	binaryLoss = binaryLoss / i;
end


function logisticLoss = logisticLossAlgo(weight, data, labels)
    % logistic loss is average hinge loss over all data points
    logisticLoss = 0;
    for i=1:size(labels)
        loss = 1 - labels(i) * dot(weight, data(i,:));
        loss = max(0, loss);
        logisticLoss = logisticLoss + loss;
    end
    logisticLoss = logisticLoss / i;
end
