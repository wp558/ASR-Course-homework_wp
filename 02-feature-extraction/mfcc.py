import librosa
import numpy as np
from scipy.fftpack import dct

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# x = np.linspace(-1,1,50)
# y1 = 2*x + 1
# y2 = x**2
#
# plt.figure()
# plt.plot(x,y1)
#
# plt.figure(num=2,figsize=(8,5))
# plt.plot(x,y2)
# plt.show()
#def plot_spectrogram(spec, note, file_name):
#    """Draw the spectrogram picture
#        :param spec: a feature_dim by num_frames array(real)
#        :param note: title of the picture
#        :param file_name: name of the file
#    """
#    fig = plt.figure(figsize=(20,5))
#    heatmap = plt.pcolor(spec)
#    fig.colorbar(mappable=heatmap)
#    plt.xlabel('Time(s)')
#    plt.ylabel(note)
#    plt.tight_layout()
#    plt_savefig(file_name)

#preemphasis config
alpha = 0.97

#Enframe config
frame_len = 400    #25ms,fs=16khz
frame_shift = 160
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

#Read wav file
#wav,fs = librosa.load('./test.wav',sr = None)

#Enframe with Hamming window function
def preemphasis(signal,coeff=alpha):
    """perform preemphasis on the input signal.
        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])
def enframe(signal,frame_len = frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift) +1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len]
        frames[i,:] = frames[i,:] * win
    return frames
def get_spectrum(frames, fft_len=fft_len):
    cFFT = np.fft.fft(frames,n=fft_len)
    valid_len = int(fft_len/2) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    print(spectrum.shape)
    return spectrum
def fbank(spectrum, fs, num_filter = num_filter, fft_len = fft_len):
    pow_frames = ((1.0/fft_len)*(spectrum**2))
    print(pow_frames.shape)
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (fs/2)/700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filter+2)
    linear_points = 700 * (10**(mel_points / 2595) - 1)
    Mel_filter = np.zeros((num_filter, int(fft_len / 2 + 1)))
    Mel_k = (linear_points/(fs/2))*(fft_len/2)
    for i in range(1, num_filter+1):
        start = int(Mel_k[i-1])
        center = int(Mel_k[i])
        end = int(Mel_k[i+1])
        for j in range(start, center):
            Mel_filter[i-1, j+1] = (j+1 - int(Mel_k[i-1])) / (int(Mel_k[i])-int(Mel_k[i-1]))
        for j in range(center, end):
            Mel_filter[i-1, j+1] = (int(Mel_k[i+1]) - (j+1)) / (int(Mel_k[i+1]) - int(Mel_k[i]))
    filter_banks = np.dot(pow_frames,Mel_filter.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    feats = 20*np.log10(filter_banks)
    return feats

def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array
    """

    feats = np.zeros((fbank.shape[0],num_mfcc))
    feats = dct(fbank,type = 2, axis=1, norm='ortho')[:, 1:(num_mfcc+1)]
    """
        FINISH by YOURSELF
    """
    return feats

def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()
def main():
    wav, fs = librosa.load('02-feature-extraction_test.wav',sr = None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum, fs)
    mfcc_feats = mfcc(fbank_feats)
    plt.imshow(mfcc_feats.T, cmap=plt.cm.jet, aspect='auto')
    plt.show()

if __name__ == '__main__':
    main()