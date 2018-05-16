import numpy as np
from scipy.signal import lfilter, hamming, stft, istft
from scipy.fftpack import fft, dct, idct
from scipy.optimize import linprog
from scipy.linalg import lstsq
import wave
import array
from scipy.io import wavfile

from .Parameters import *
from .audio_utilities import reconstruct_signal_griffin_lim


def read_wavfile(wav_filename, channel='mixed'):
    """Read a WAV audio file and returns a 1D numpy array of time series samples.

    Args:
        wav_filename (path): The filename of the WAV audio signal to read.
        channel (str): The audio channel selected.
    Outputs:
        samples (1D numpy array): The 1D-array of audio samples read.
    """
    assert channel in {'instrumental', 'vocals', 'mixed'}
    wav = wavfile.read(wav_filename)[1]
    if channel == 'instrumental':
        samples = wav[:,0]
    elif channel == 'vocals':
        samples = wav[:,1]
    else:
        samples = np.sum(wav, axis=-1)
    return samples


def wav_to_mfcc(wav_filename, channel='mixed'):
    """Convert a WAV audio signal to MFCC and Mel Spectrum 2D numpy arrays.

    Args:
        wav_filename (path): The WAV audio signal to convert.
        channel (str): The audio channel selected.
    Outputs:
        mfccs (2D numpy array): The MFCCs [n_windows, n_ceps] of the selected mono audio signal.
        mels (2D numpy array): The Mel Spectrum [n_windows, n_mels] of the selected mono audio signal.
        mweights (2D numpy array): The Mel Spectrum weights used for Griffin-Lim reconstruction.
    """
    samples = read_wavfile(wav_filename, channel)
    mfccs, mspec, mweights = mfcc(samples)
    return mfccs, mspec, mweights


def mfcc_to_wav(mfccs, mel_weights):
    """Convert an MFCC 2D numpy array to a time series audio signal.

    Args:
        mfccs (2D numpy array): The MFCCs [nfft//2 + 1, n_windows] to convert.
        mel_weights (2D numpy array): The Mel Spectrum weights used for Griffin-Lim reconstruction.
    Outputs:
        wav (1D numpy array):  The reconstructed mono audio signal.
    """
    melspec = np.exp(icepstrum(mfccs))
    # option A: l2-norm
    pow_spec = imelspectrum_l2(melspec, mel_weights)
    # option B: l1-norm
    # mag_spec = imelspectrum_l1(rec_mspec, mel_weights)
    mag_spec = np.sqrt(pow_spec)[:, :Preprocessing.NFFT//2 + 1]
    wav = reconstruct_signal_griffin_lim(mag_spec,
                                         Preprocessing.NFFT,
                                         Preprocessing.WINSHIFT,
                                         Postprocessing.GRIFFIN_LIM_ITE)
    assert not np.isnan(wav).any()
    # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    max_sample = np.max(abs(wav))
    if max_sample > 1.0:
        wav = wav / max_sample
    return wav


def mspec_to_wav(logmspec, mel_weights):
    """Convert a (log) Mel Spectrum 2D numpy array to a time series audio signal.

    Args:
        logmspec (2D numpy array): The (log) Mel Spectrum [n_mels, n_windows] to convert.
        mel_weights (2D numpy array): The Mel Spectrum weights used for Griffin-Lim reconstruction.
    Outputs:
        wav (1D numpy array):  The reconstructed mono audio signal.
    """
    # option A: l2-norm
    pow_spec = imelspectrum_l2(np.exp(logmspec), mel_weights)
    # option B: l1-norm
    # mag_spec = imelspectrum_l1(rec_mspec, mel_weights)
    mag_spec = np.sqrt(pow_spec)[:, :Preprocessing.NFFT//2 + 1]
    wav = reconstruct_signal_griffin_lim(mag_spec,
                                         Preprocessing.NFFT,
                                         Preprocessing.WINSHIFT,
                                         Postprocessing.GRIFFIN_LIM_ITE)
    assert not np.isnan(wav).any()
    # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    max_sample = np.max(abs(wav))
    if max_sample > 1.0:
        wav = wav / max_sample
    return wav


def mfcc(samples, winlen=Preprocessing.WINLEN, winshift=Preprocessing.WINSHIFT,
            preempcoeff=Preprocessing.PREEMPCOEFF, nfft=Preprocessing.NFFT, nceps=Preprocessing.NCEPS,
            samplingrate=Preprocessing.FS, liftercoeff=22, with_lifter=False):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec, mweights = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    if with_lifter:
        return lifter(ceps, liftercoeff), mspec, mweights
    else:
        return ceps, mspec, mweights


def wav_to_stft(wav_filename, channel='mixed'):
    """Convert a WAV audio signal to its STFT (magnitude-only!) 2D numpy array.

    Args:
        wav_filename (path): The filename of the WAV audio signal to convert.
        channel (str): The audio channel selected.
    Outputs:
        Zxx (2D numpy array): The STFT [nfft//2 + 1, n_windows] of the selected mono audio signal.
    """
    samples = read_wavfile(wav_filename, channel)
    freqs, times, Zxx = stft(samples,
                             fs=Preprocessing.FS,
                             window=Preprocessing.WINDOW,
                             nperseg=Preprocessing.WINLEN,
                             noverlap=Preprocessing.WINSHIFT,
                             nfft=Preprocessing.NFFT,
                             detrend=Preprocessing.DETREND,
                             return_onesided=Preprocessing.ONESIDED,
                             boundary=Preprocessing.BOUNDARY,
                             padded=Preprocessing.PADDED,
                             axis=-1)
    return np.absolute(Zxx).T


def stft_to_wav(Zxx):
    """Convert an STFT (magnitude-only!) 2D numpy array to a time series audio signal.

    Args:
        Zxx (2D numpy array): The STFT [nfft//2 + 1, n_windows] to convert.
    Outputs:
        wav (1D numpy array):  The reconstructed mono audio signal.
    """
    times, wav = istft(Zxx,
                       fs=Preprocessing.FS,
                       window=Preprocessing.WINDOW,
                       nperseg=Preprocessing.WINLEN,
                       noverlap=Preprocessing.WINSHIFT,
                       nfft=Preprocessing.NFFT,
                       input_onesided=Preprocessing.ONESIDED,
                       boundary=Preprocessing.BOUNDARY,
                       time_axis=-1,
                       freq_axis=-2)
    assert not np.isnan(wav).any()
    # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    max_sample = np.max(abs(wav))
    if max_sample > 1.0:
        wav = wav / max_sample
    return wav


def save_audio_to_file(x, filename='out.wav', sample_rate=Preprocessing.FS):
    """Save a mono signal to a file.

    Args:
        x (1D numpy array): The audio signal to save. The signal values should be in the range [-1.0, 1.0].
        filename (path): Name of the file to save.
        sample_rate (int): The sample rate of the signal, in Hz.
    """
    x_max = np.max(abs(x))
    assert x_max <= 1.0, 'Input audio value is out of range. Should be in the range [-1.0, 1.0].'
    x = x*32767.0
    data = array.array('h')
    for i in range(len(x)):
        cur_samp = int(round(x[i]))
        data.append(cur_samp)
    f = wave.open(filename, 'w')
    f.setparams((1, 2, sample_rate, 0, 'NONE', 'Uncompressed'))
    f.writeframes(data.tostring())
    f.close()


def save_audio_to_file2(x, filename='out.wav', sample_rate=Preprocessing.FS):
    wavfile.write(filename, sample_rate, x)


def dither(samples, level=1.0):
    """
    Applies dithering to the samples. Adds Gaussian noise to the samples to avoid numerical
        errors in the subsequent FFT calculations.

        samples: array of speech samples
        level: decides the amount of dithering (see code for details)

    Returns:
        array of dithered samples (same shape as samples)
    """
    return samples + level*np.random.normal(0,1, samples.shape)


def hz2mel(f):
    """Convert an array of frequency in Hz into mel."""
    return 1127.01048 * np.log(f/700 +1)

def trfbank(fs, nfft, lowfreq=133.33, linsc=200/3., logsc=1.0711703, nlinfilt=13, nlogfilt=27, equalareas=False):
    """Compute triangular filterbank for MFCC computation.
    Inputs:
    fs:         sampling frequency (rate)
    nfft:       length of the fft
    lowfreq:    frequency of the lowest filter
    linsc:      scale for the linear filters
    logsc:      scale for the logaritmic filters
    nlinfilt:   number of linear filters
    nlogfilt:   number of log filters
    Outputs:
    res:  array with shape [N, nfft], with filter amplitudes for each column.
            (N=nlinfilt+nlogfilt)
    From scikits.talkbox"""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    if equalareas:
        heights = np.ones(nfilt)
    else:
        heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank


def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    ls = []
    for i in range(0, len(samples)//winlen*winlen, winshift):
        if i+winlen > len(samples):
            break
        ls.append(samples[i:i+winlen])

    return np.array(ls)

def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    # Attention a mettre une virgule dans la creation du tableau b !!!!
    return lfilter([1, -p], [1], input)


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    N, M = input.shape
    window = hamming(M, sym=False)
    return (input * window)

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    freq = fft(input, nfft)
    return freq.real**2 + freq.imag**2


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbanks)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    N, nfft = input.shape
    flt = trfbank(samplingrate, nfft)
    return np.log(np.dot(input, flt.transpose())), flt


def cepstrum(input, nceps, all=False):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    if all:
        return dct(input, norm='ortho')
    else:
        return dct(input, norm='ortho')[:, 0:nceps]


def lifter(mfcc, lifter=22):
    """
    Applies liftering to improve the relative range of MFCC coefficients.
       mfcc: NxM matrix where N is the number of frames and M the number of MFCC coefficients
       lifter: lifering coefficient
    Returns:
       NxM array with lifeterd coefficients
    """
    nframes, nceps = mfcc.shape
    cepwin = 1.0 + lifter/2.0 * np.sin(np.pi * np.arange(nceps) / lifter)
    return np.multiply(mfcc, np.tile(cepwin, nframes).reshape((nframes,nceps)))


def icepstrum(inp):
    return idct(inp, norm='ortho')


def imelspectrum_l2(inp, mel_weights):
    # print(np.isnan(inp).any(), np.isnan(mel_weights).any())
    # print(np.argwhere(np.isnan(inp)))
    # print(inp.shape, mel_weights[:inp.shape[1]+1,:].shape)
    return abs(lstsq(mel_weights[:inp.shape[1],:], inp.T)[0].T)

def imelspectrum_l1(inp, mel_weights): #NOT WORKING
    return linprog(inp, method='interior-point')


def denoise(y, b=[1.0/Postprocessing.DENOISE_FACTOR]*Postprocessing.DENOISE_FACTOR, a=1):
    return lfilter(b, a, y)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def dtw(x, y, dist=euclidean_distance):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path through AD

    Note that you only need to define the first output for this exercise.
    """
    LD = distance_matrix(y, x)
    AD = np.zeros_like(LD)
    AD[0, 0] = LD[0, 0]
    for i in range(1, len(x)):
        AD[0, i] = LD[0, i] + AD[0, i-1]
    for i in range(1, len(y)):
        AD[i, 0] = LD[i, 0] + AD[i-1, 0]
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            AD[i, j] = min(AD[i-1, j-1], AD[i-1, j], AD[i, j-1]) + LD[i, j]
    path = [[len(x)-1, len(y)-1]]
    cost = 0
    i = len(y)-1
    j = len(x)-1
    while i > 0 and j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            if AD[i-1, j] == min(AD[i-1, j-1], AD[i-1, j], AD[i, j-1]):
                i = i - 1
            elif AD[i, j-1] == min(AD[i-1, j-1], AD[i-1, j], AD[i, j-1]):
                j = j-1
            else:
                i = i - 1
                j = j - 1
        path.append([j, i])
    path.append([0, 0])
    for [p1, p0] in path:
        cost = cost + LD[p0, p1]
    return cost/(len(x)+len(y)), LD, AD, path
