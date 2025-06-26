% M1_feature_extraction.m - Extract PSD features from LFP data

% Clear workspace and command window
clc; clear;

% Add utility functions to MATLAB path
addpath(genpath("./matlab_script/utils"));

% Configuration parameters
windowSize = 10;             % Window size in seconds
sampleRate = 1000;           % Data sampling rate in Hz
% Frequency bands for analysis: Delta, Theta, Alpha, Beta
frequencyBands = {[1,2]; [2,3]; [3,4]; [4,5]};
featureType = "psd";         % Feature type (Power Spectral Density)

% Define output directory for saving results
outputDir = "./dataset/feature/";
plotDir = "./result/plot_png/";

% Create output directories if they don't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
if ~exist(plotDir, 'dir')
    mkdir(plotDir);
end

% Path to preprocessed data
dataDir = "./dataset/preprocessed_data";
dataFiles = dir(dataDir);

% Process each data file
for fileIdx = 3:size(dataFiles, 1)  % Start from 3 to skip '.' and '..' directories
    % Get current filename
    currentFileName = dataFiles(fileIdx).name;
    
    % Load raw data
    rawData = load(fullfile(dataDir, currentFileName));
    data = rawData.preprocessedData;
    
    % Initialize feature container
    psdFeatures = data(1:end, 1);
    
    % Extract EEG data and channel names
    lfpData = data{2};
    channelNames = data{1};
    
    % Calculate PSD features for different frequency bands
    % psdFeatures - Cell array containing:
    % {1} - Matrix of band powers for each time window
    % {2} - Full PSD values for each time window
    % {3} - Frequency vector corresponding to PSD values
    resultFeatures = get_psd(lfpData, sampleRate, windowSize, frequencyBands);
    
    % Store the extracted features for each frequency band
    for bandIdx = 1:size(resultFeatures,1)
        psdFeatures{bandIdx+1} = resultFeatures{bandIdx};
    end
    
    % Log progress
    fprintf("Processed file: %s\n", currentFileName);
    
    % Prepare output filename (remove .mat extension from original filename)
    baseFileName = currentFileName(1:end-4);
    outputFilePath = sprintf("%s/%s_%s.mat", outputDir, baseFileName, featureType);
    
    % Save extracted features
    features = psdFeatures;
    save(outputFilePath, 'features');
    
    % Create visualization of PSD results
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot average PSD across all channels
    subplot(2,1,1);
    freqVec = resultFeatures{end};  % Frequency vector from the last element
    avgPSD = mean(resultFeatures{end-1}, 2);  % Average PSD across all windows
    plot(freqVec, avgPSD, 'LineWidth', 2);
    grid on;
    title(['Average PSD - ', baseFileName], 'Interpreter', 'none');
    xlabel('Frequency (Hz)');
    ylabel('Power Spectral Density (μV²/Hz)');
    xlim([0, 30]);  % Limit x-axis to 0-30 Hz for better visualization
    
    % Plot band power distribution as violin plot with scatter points
    subplot(2,1,2);
    bandPowers = cell2mat(resultFeatures(1));
    bandNames = {'Delta', 'Theta', 'Alpha', 'Beta'};
    
    % Create violin plots
    violinplot(bandPowers, bandNames, 'ShowData', true, 'ViolinAlpha', 0.5);
    grid on;
    title(['Band Power Distribution - ', baseFileName], 'Interpreter', 'none');
    ylabel('Power (μV²)');
    % If violinplot function is not available in your MATLAB version, 
    % you might need to download it from MATLAB File Exchange or use:
    % https://github.com/bastibe/Violinplot-Matlab
    % Save the figure as PNG
    plotFilePath = sprintf("%s/M1_feature_extracttion_%s_%s.png", plotDir, baseFileName, featureType);
    saveas(gcf, plotFilePath);
    close(gcf);
end
