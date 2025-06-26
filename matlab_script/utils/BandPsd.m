function [bandPower, powerSpectrum, frequencies] = BandPsd(signalData, samplingRate, frequencyBands)
%% BANDPSD Calculate the power spectral density in specified frequency bands
% Computes power in different frequency bands for a time series signal
%
% Inputs:
%   signalData:       Time series data (LFP or other signal) - must be a vector
%   samplingRate:     Sampling rate in Hz
%   frequencyBands:   n×1 cell array where each cell contains [lowFreq, highFreq]
%                     Example: {[1,4]; [5,8]; [9,12]; [13,30]; [31,70]; [71,150]}
%
% Outputs:
%   bandPower:        Power in each frequency band (in dB)
%   powerSpectrum:    Full power spectral density
%   frequencies:      Frequency vector corresponding to powerSpectrum
%%
    % Input validation
    if ~isvector(signalData)
        error('Input signal must be a vector');
    end

    % Initialize output array
    numBands = size(frequencyBands, 1);
    bandPower = zeros(numBands, 1);
    
    % Set window parameters for Welch's method
    windowLength = min(round(samplingRate), length(signalData));
    overlap = [];  % Default overlap (50%)
    fftLength = windowLength;
    
    % Calculate power spectral density using Welch's method
    [powerSpectrum, frequencies] = pwelch(signalData, windowLength, overlap, fftLength, samplingRate);
    
    % Calculate power in each frequency band
    for bandIdx = 1:numBands
        lowerBound = frequencyBands{bandIdx}(1);
        upperBound = frequencyBands{bandIdx}(2);
        
        % Identify frequencies within the specified band
        bandMask = (frequencies >= lowerBound & frequencies <= upperBound);
        
        % Calculate mean power in band and convert to dB
        bandPower(bandIdx) = pow2db(mean(powerSpectrum(bandMask)));
    end
end