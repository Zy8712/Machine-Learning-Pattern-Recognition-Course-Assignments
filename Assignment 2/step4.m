% Load the Dataset to be worked on
dataset_input = load('fg_inputs.txt'); % load input data points
dataset_output = load('fg_outputs.txt'); % load corresponding binary outputs

% Classify Data Points from Dataset
dataset_input(:,4) = dataset_output; % set input points equal to output results
positive_points = dataset_input(dataset_input(:,4) == 1,:); % extract positive points
negative_points = dataset_input(dataset_input(:,4) == -1,:); % extract negative points

figure; % create a new figure window with default property values

% Plot Positive & Negative Data Points
plot3(positive_points(:,1),positive_points(:,2),positive_points(:,3),'+'); % plot positive data points
hold on; % retains plots in the current axes so new plots do not delete it
plot3(negative_points(:,1),negative_points(:,2),negative_points(:,3),'o'); % plot negative data points
hold on; % retains plots in the current axes so new plots do not delete it

grid on; % display the major grid lines for the current axes

% Plot Linear Predictor/Separator
%x = linspace(-4,4); % x coordinates
%y = linspace(-4,4); % y coordinates
%plot(x,y); % plot line on grid

% Piece the figure together
ax = gca; % returns the current axes in the current figure (gca creates a Cartesian axes object)
ax.XAxisLocation = 'origin'; % set axes center point to (0,0) origin
ax.YAxisLocation = 'origin'; % set axes center point to (0,0) origin

% Set Text in Graph
title('Plotting and Classification of Data Points'); % title the figure/plot
legend('Positive Classification', 'Negative Classification', 'Linear Predictor/Separator'); % label legend

% Set Plot Limits
%xlim([-4,4]); % set domain of possible values to be displayed on plot
%ylim([-4,4]); % set range of possible values to be displayed on plot