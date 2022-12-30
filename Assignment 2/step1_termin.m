% Load the Dataset to be worked on
dataset_input = load('6pointsinputs.txt'); % load input data points
dataset_output = load('6pointsoutputs.txt'); % load corresponding binary outputs

% Output Information
disp('Input Data Points from 6pointsinputs.txt:'); % print statement
disp(dataset_input); % print out input data point matrix
disp('Binary Outputs from 6pointsoutputs.txt:'); % print statement
disp(dataset_output); % print out corresponding binary outputs (i.e. label vector)

% Loop through to prove that independently of the order in which the perceptron
% algorithm gets to see these data points, the algorithm will terminate after one update
for i = 1:200 % Number of passes was never defined, thus we use 100 as stated in step 2
   disp('Loop Number:'); % print statement
   disp(i); % Output the loop number
   % Call the function that returns the number of updates before termination
   numberOfUpdates = proveTermination(dataset_input, dataset_output); % Equal to returned value
   disp('Number of Updates:'); % print statement
   disp(numberOfUpdates); % Display the number of updates that occured on the latest pass
end

% Function that returns the number of updates before termination.
function numberOfUpdates = proveTermination(dataset_input,dataset_output)
    % Size returns the number of rows and columns of the matrix and
    % assigns to corresponding values rows and cols
    [rowNum, colNum] = size(dataset_input); % assignment statement
    weightVector = zeros(1,colNum); % declare a 1x2 matrix consisting of zeros
    updateNum = 0; % declare a counter for number of updates
    marker = -1; % declare a marker indicating when more training is needed
    j = 1; % declare counter j

    % Establish our matrix of values which contains both the input and
    % corresponding output values. Matrix will have the same number of rows
    % as the input (and output) and the number of columns will be equal to
    % the input plus the output.
    matrixM = zeros(rowNum, colNum+1); % Create a matrix of zeros.
    matrixM(:,1:2) = dataset_input; % Set the first two columns equal to values from the input matrix.
    matrixM(:,3) = dataset_output; % Set the last column equal to values from the output matrix.
    matrixM = matrixM(randperm(size(matrixM, 1)), :); % Randomize order of values.
    
    % Take the randomized points and store their respective inputs and
    % outputs separately.
    dataset_input_mod = matrixM(:,1:2); % Matrix for input data points.
    dataset_output_mod = matrixM(:,3); % Matrix for corresponding outputs.

    % Output Information
    disp('Shuffled Input Data Points:'); % print statement
    disp(dataset_input_mod); % print out randomized input data point matrix
    disp('Shuffled Corresponding Binary Outputs:'); % print statement
    disp(dataset_output_mod); % print out corresponding randomized binary outputs (i.e. label vector)

    % While loop. Loops until conditions are no longer satisfied.
    while j <= 100 && marker == -1
        marker = 0; % unmark marker
        k = 1; % declare counter k
        while k <= rowNum % cycle through the six data points
            % Perform calculations which we'll use to determine if an
            % update is warranted and if more training is required.
            signValue = dot(weightVector, dataset_input_mod(k,:)) * dataset_output_mod(k);
            if signValue <= 0 % if negative or equal to 0
                weightVector = weightVector + (dataset_input_mod(k,:) * dataset_output_mod(k));
                updateNum = updateNum + 1; % increment update counter
                marker = -1; % remark marker
            end
            k = k + 1; % increment counter
        end
        j = j + 1; % increment counter
    end
    numberOfUpdates = updateNum; % set the value to be returned equal to updateNum
end