function [psdFeatures] = get_psd(data, sampleRate, windowSize, freqList)
    % GET_PSD Calculates power spectral density features from time-series data
    %
    % Inputs:
    %   data - Input time series signal (column vector)
    %   sampleRate - Sampling rate of the signal in Hz
    %   windowSize - Size of the window in seconds
    %   freqList - List of frequency bands for which to calculate PSD [Nx2] matrix
    %              where each row is [lowFreq, highFreq]
    %
    % Outputs:
    %   psdFeatures - Cell array containing:
    %       {1} - Matrix of band powers for each time window
    %       {2} - Full PSD values for each time window
    %       {3} - Frequency vector corresponding to PSD values
    
    % Initialize output cell array
    psdFeatures = cell(3, 1);
    numBands = size(freqList, 1);
    
    % Set window parameters
    overlapRatio = 0.5;  % 50% overlap between consecutive windows
    windowLength = round(windowSize * sampleRate);
    strideLength = round((1 - overlapRatio) * windowSize * sampleRate);
    
    % Calculate number of windows
    totalSamples = size(data, 1);
    numWindows = floor((totalSamples - windowLength) / strideLength) + 1;
    
    % Initialize matrices to store results
    bandPowerMatrix = zeros(numWindows, numBands);
    signalSegmentMatrix = zeros(windowLength, numWindows);
    
    % Extract signal segments using sliding window
    for iWindow = 1:numWindows
        segmentIndices = (1:windowLength) + (iWindow - 1) * strideLength;
        signalSegmentMatrix(:, iWindow) = data(segmentIndices);
    end
    
    % Calculate PSD for each window
    pxxList = [];
    for iWindow = 1:numWindows
        currentSegment = signalSegmentMatrix(:, iWindow);
        [bandPower, pxx, freqVector] = BandPsd(currentSegment, sampleRate, freqList);
        
        % Store results
        pxxList = [pxxList, pxx];
        bandPowerMatrix(iWindow, :) = bandPower;
    end
    
    % Populate output structure
    psdFeatures{1} = bandPowerMatrix;
    psdFeatures{2} = pxxList;
    psdFeatures{3} = freqVector;
end