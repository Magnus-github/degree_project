import torch.nn as nn
import torch
import pywt
import numpy as np


def AvgPool(kernel_size: int):
    return nn.AvgPool1d(kernel_size)


def MaxPool(kernel_size: int):
    return nn.MaxPool1d(kernel_size)


class Voting:
    def __init__(self, dummy):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __call__(self, x):
        x = x.to('cpu')
        x = torch.mode(x, dim=-1).values
        return x.to(self.device)


class HistogramFeatures:
    def __init__(self, num_bins: int = 10) -> None:
        self.num_bins = num_bins

    def __call__(self, x):
        B, T, C, J = x.shape
        reshaped_x = x.reshape(T, B*C*J)
    
        # Compute histograms along each channel and joint dimension
        histograms = []
        for i in range(reshaped_x.size(1)):
            channel_data = reshaped_x[:, i]
            hist = torch.histc(channel_data, bins=self.num_bins, min=channel_data.min(), max=channel_data.max()) / channel_data.size(0)
            histograms.append(hist)
        
        # Stack histograms along the channel dimension
        histograms = torch.stack(histograms, dim=0)

        histograms = histograms.reshape(B, self.num_bins, C, J)

        return histograms
    

class WaveletTransform:
    def __init__(self, wavelet: str = 'morl', level: int = 300) -> None:
        self.wavelet = wavelet
        self.level = level

    def __call__(self, x):
        B, T, C, J = x.shape
        reshaped_x = x.reshape(B*C*J, T).numpy()
        spectrograms = np.zeros((B*C*J, self.level, T))
    
        # Iterate over each time series
        for i in range(B*C*J):
            # Perform continuous wavelet transform
            coeffs, _ = pywt.cwt(reshaped_x[i], torch.arange(1, self.level + 1), self.wavelet)
            
            # Calculate power spectral density
            power = (abs(coeffs)) ** 2
            
            # Downsample to obtain spectrogram
            spectrograms[i] = power
        
        return torch.tensor(spectrograms, dtype=torch.float32)
    

# def wavelet_tf(x):
#     # x.shape: [T, 1]
#     s = 
#     wavelet = 


    
if __name__=="__main__":
    x = torch.randn(3, 240, 7, 18)
    wv_tf = WaveletTransform()
    spectrograms = wv_tf(x)
    print(spectrograms.shape)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    for i in range(18):
        plt.imshow(spectrograms[i].numpy(), aspect='auto', cmap='plasma', origin='lower', interpolation='nearest')
        plt.colorbar(label='Power')
        plt.title(f'Spectrogram for Time Series {i}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()


