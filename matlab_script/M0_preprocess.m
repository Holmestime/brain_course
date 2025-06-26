% M0_preprocess.m - preprocess the LFP data

% Clear workspace and command window
clc; clear;

%% Data Selection and Initialization
% Set default folder for raw data files
defaultFolderPath = fullfile('dataset', 'raw_data');

% Get list of .mat files in the default folder
matFileList = dir(fullfile(defaultFolderPath, '*.mat'));

% Initialize cell array to store preprocessed data
% First cell contains the name, second contains the preprocessed signal
preprocessedData = cell(1, 2);

% Check if any .mat files exist in the selected folder
if isempty(matFileList)
    disp('No .mat files found in the selected folder.');
    return;
end

%% Output Directory Setup
% Create directory for preprocessed data
outputDirPath = fullfile('dataset', 'preprocessed_data');
if ~exist(outputDirPath, 'dir')
    mkdir(outputDirPath);
end

% Create directory for visualization figures
visualDirPath = fullfile('result', 'plot_png');
if ~exist(visualDirPath, 'dir')
    mkdir(visualDirPath);
end

%% Process Each Data File
% Loop through each file
for fileIdx = 1:length(matFileList)
    % Load current data file
    currentFile = matFileList(fileIdx);
    fprintf('Processing file %d/%d: %s\n', fileIdx, length(matFileList), currentFile.name);
    
    % Extract file name and load data
    fileName = currentFile.name;
    filePrefix = fileName(1:min(10, length(fileName)));
    rawData = load(fullfile(defaultFolderPath, fileName)).data;
    
    % Process each row in the raw data
    for rowIdx = 1:size(rawData, 1)
        % Extract current signal data and name
        currentSignal = rawData{rowIdx, 5};
        currentName = rawData{rowIdx, 1}{1};
        
        % Preprocess the signal and get visualization data
        [processedSignal, visualData] = preprocessSignal(currentSignal);
        
        % Store processed data
        preprocessedData{1} = currentName;
        preprocessedData{2} = processedSignal;
        
        % Save the processed data
        outputFilePath = fullfile(outputDirPath, sprintf('%s.mat', currentName));
        save(outputFilePath, 'preprocessedData');
        
        % Generate and save visualization
        visualizationPath = fullfile(visualDirPath, sprintf('M0_preprocess_visualization_%s.png', currentName));
        plotAndSaveResults(visualData, currentName, visualizationPath);
        
        fprintf('Finished processing file %d/%d: %s\n', fileIdx, length(matFileList), fileName);
    end
end

%% Signal Preprocessing Function
function [filteredData, visualData] = preprocessSignal(inputSignal)
    %PREPROCESSSIGNAL Apply filters to clean the raw signal
    %   This function applies a sequence of filters to the input signal:
    %   1. A low-pass filter to remove high frequency noise (>30 Hz)
    %   2. A high-pass filter to remove DC offset and low-frequency drift (<1 Hz) 
    %   3. Notch filters to remove power line interference and its harmonics
    %
    %   Parameters:
    %       inputSignal: Raw signal data to be filtered
    %
    %   Returns:
    %       filteredData: Processed signal after applying all filters
    %       visualData: Structure containing intermediate processing results
    
    % Define filter parameters
    sampleRate = 1000;       % Hz
    filterOrder = 12;        % Filter order for all filters
    notchFrequency = 50;     % Hz - center frequency to remove
    notchBandwidth = 2;      % Hz - bandwidth around center frequency
    harmonicCount = 1;       % Number of harmonics to filter
    
    % Store original signal for visualization
    visualData.raw = inputSignal;
    visualData.sampleRate = sampleRate;
    
    % Design low-pass filter
    lowpassFilter = designfilt('lowpassiir', ...
                               'FilterOrder', filterOrder, ...
                               'HalfPowerFrequency', 25, ...
                               'DesignMethod', 'butter', ...
                               'SampleRate', sampleRate);
    
    % Apply low-pass filter
    tempFilteredData = filter(lowpassFilter, inputSignal);
    visualData.afterLowpass = tempFilteredData;
    
    % Design high-pass filter
    highpassFilter = designfilt('highpassiir', ...
                                'FilterOrder', filterOrder, ...
                                'HalfPowerFrequency', 20, ...
                                'DesignMethod', 'butter', ...
                                'SampleRate', sampleRate); 
    
    % Apply high-pass filter
    filteredData = filter(highpassFilter, tempFilteredData);
    visualData.afterHighpass = filteredData;
    
    % Apply notch filters for specified harmonics
    for harmonicIdx = 1:harmonicCount
        % Calculate current harmonic frequency
        currentNotchFreq = notchFrequency * harmonicIdx;
        
        % Design notch filter for current harmonic
        notchFilter = designfilt('bandstopiir', ...
                                'FilterOrder', filterOrder, ...
                                'HalfPowerFrequency1', currentNotchFreq - notchBandwidth, ...
                                'HalfPowerFrequency2', currentNotchFreq + notchBandwidth, ...
                                'DesignMethod', 'butter', ...
                                'SampleRate', sampleRate);
        
        % Apply notch filter
        filteredData = filter(notchFilter, filteredData);
    end
    
    visualData.final = filteredData;
end

%% Visualization Function
function plotAndSaveResults(visualData, signalName, savePath)
    % Create time vector
    
    signalLength = length(visualData.raw);
    timeVector = (0:signalLength-1) / visualData.sampleRate;
    
    % Parameters for pwelch
    % Use window length that represents 1 second of data for 1000 Hz sampling rate
    window = hamming(visualData.sampleRate);  % 1 second window at 1000 Hz
    noverlap = floor(length(window)/2);  % 50% overlap
    nfft = min(round(visualData.sampleRate),length(visualData.raw));
    
    % Create a figure with 4 subplots
    figure('Position', [100, 100, 1000, 800]);
    
    % Plot 1: Raw signal in time domain
    subplot(4, 2, 1);
    plot(timeVector, visualData.raw);
    title('Raw Signal (Time Domain)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Plot 2: Raw signal in frequency domain
    subplot(4, 2, 2);
    [pxx, f] = pwelch(visualData.raw, window, noverlap, nfft, visualData.sampleRate);
    plot(f, 10*log10(pxx));
    title('Raw Signal (Frequency Domain)');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0, 100]); % Limit view to 0-100 Hz for better visualization
    grid on;
    
    % Plot 3: After lowpass filter in time domain
    subplot(4, 2, 3);
    plot(timeVector, visualData.afterLowpass);
    title('After Lowpass Filter (Time Domain)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Plot 4: After lowpass filter in frequency domain
    subplot(4, 2, 4);
    [pxx, f] = pwelch(visualData.afterLowpass, window, noverlap, nfft, visualData.sampleRate);
    plot(f, 10*log10(pxx));
    title('After Lowpass Filter (Frequency Domain)');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0, 100]);
    grid on;
    
    % Plot 5: After highpass filter in time domain
    subplot(4, 2, 5);
    plot(timeVector, visualData.afterHighpass);
    title('After Highpass Filter (Time Domain)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Plot 6: After highpass filter in frequency domain
    subplot(4, 2, 6);
    [pxx, f] = pwelch(visualData.afterHighpass, window, noverlap, nfft, visualData.sampleRate);
    plot(f, 10*log10(pxx));
    title('After Highpass Filter (Frequency Domain)');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0, 100]);
    grid on;
    
    % Plot 7: Final filtered signal in time domain
    subplot(4, 2, 7);
    plot(timeVector, visualData.final);
    title('Final Filtered Signal (Time Domain)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Plot 8: Final filtered signal in frequency domain
    subplot(4, 2, 8);
    [pxx, f] = pwelch(visualData.final, window, noverlap, nfft, visualData.sampleRate);
    plot(f, 10*log10(pxx));
    title('Final Filtered Signal (Frequency Domain)');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0, 100]);
    grid on;
    
    % Set overall title for the figure
    sgtitle(['Signal Processing Results: ', signalName], 'FontSize', 14, 'FontWeight', 'bold');
    
    % Make the figure look nicer
    set(gcf, 'Color', 'white');
    
    % Save the figure
    saveas(gcf, savePath);
    
    % Close the figure
    close(gcf);
end
