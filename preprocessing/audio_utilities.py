import numpy as np


def hz_to_mel(f_hz):
    """Convert Hz to mel scale.

    This uses the formula from O'Shaugnessy's book.
    Args:
        f_hz (float): The value in Hz.

    Returns:
        The value in mels.
    """
    return 2595*np.log10(1.0 + f_hz/700.0)


def mel_to_hz(m_mel):
    """Convert mel scale to Hz.

    This uses the formula from O'Shaugnessy's book.
    Args:
        m_mel (float): The value in mels

    Returns:
        The value in Hz
    """
    return 700*(10**(m_mel/2595) - 1.0)


def fft_bin_to_hz(n_bin, sample_rate_hz, fft_size):
    """Convert FFT bin index to frequency in Hz.

    Args:
        n_bin (int or float): The FFT bin index.
        sample_rate_hz (int or float): The sample rate in Hz.
        fft_size (int or float): The FFT size.

    Returns:
        The value in Hz.
    """
    n_bin = float(n_bin)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    return n_bin*sample_rate_hz/(2.0*fft_size)


def hz_to_fft_bin(f_hz, sample_rate_hz, fft_size):
    """Convert frequency in Hz to FFT bin index.

    Args:
        f_hz (int or float): The frequency in Hz.
        sample_rate_hz (int or float): The sample rate in Hz.
        fft_size (int or float): The FFT size.

    Returns:
        The FFT bin index as an int.
    """
    f_hz = float(f_hz)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    fft_bin = int(np.round((f_hz*2.0*fft_size/sample_rate_hz)))
    if fft_bin >= fft_size:
        fft_bin = fft_size-1
    return fft_bin


def make_mel_filterbank(min_freq_hz, max_freq_hz, mel_bin_count,
                        linear_bin_count, sample_rate_hz):
    """Create a mel filterbank matrix.

    Create and return a mel filterbank matrix `filterbank` of shape (`mel_bin_count`,
    `linear_bin_couont`). The `filterbank` matrix can be used to transform a
    (linear scale) spectrum or spectrogram into a mel scale spectrum or
    spectrogram as follows:

    `mel_scale_spectrum` = `filterbank`*'linear_scale_spectrum'

    where linear_scale_spectrum' is a shape (`linear_bin_count`, `m`) and
    `mel_scale_spectrum` is shape ('mel_bin_count', `m`) where `m` is the number
    of spectral time slices.

    Likewise, the reverse-direction transform can be performed as:

    'linear_scale_spectrum' = filterbank.T`*`mel_scale_spectrum`

    Note that the process of converting to mel scale and then back to linear
    scale is lossy.

    This function computes the mel-spaced filters such that each filter is triangular
    (in linear frequency) with response 1 at the center frequency and decreases linearly
    to 0 upon reaching an adjacent filter's center frequency. Note that any two adjacent
    filters will overlap having a response of 0.5 at the mean frequency of their
    respective center frequencies.

    Args:
        min_freq_hz (float): The frequency in Hz corresponding to the lowest
            mel scale bin.
        max_freq_hz (flloat): The frequency in Hz corresponding to the highest
            mel scale bin.
        mel_bin_count (int): The number of mel scale bins.
        linear_bin_count (int): The number of linear scale (fft) bins.
        sample_rate_hz (float): The sample rate in Hz.

    Returns:
        The mel filterbank matrix as an 2-dim Numpy array.
    """
    min_mels = hz_to_mel(min_freq_hz)
    max_mels = hz_to_mel(max_freq_hz)
    # Create mel_bin_count linearly spaced values between these extreme mel values.
    mel_lin_spaced = np.linspace(min_mels, max_mels, num=mel_bin_count)
    # Map each of these mel values back into linear frequency (Hz).
    center_frequencies_hz = np.array([mel_to_hz(n) for n in mel_lin_spaced])
    mels_per_bin = float(max_mels - min_mels)/float(mel_bin_count - 1)
    mels_start = min_mels - mels_per_bin
    hz_start = mel_to_hz(mels_start)
    fft_bin_start = hz_to_fft_bin(hz_start, sample_rate_hz, linear_bin_count)
    #print('fft_bin_start: ', fft_bin_start)
    mels_end = max_mels + mels_per_bin
    hz_stop = mel_to_hz(mels_end)
    fft_bin_stop = hz_to_fft_bin(hz_stop, sample_rate_hz, linear_bin_count)
    #print('fft_bin_stop: ', fft_bin_stop)
    # Map each center frequency to the closest fft bin index.
    linear_bin_indices = np.array([hz_to_fft_bin(f_hz, sample_rate_hz, linear_bin_count) for f_hz in center_frequencies_hz])
    # Create filterbank matrix.
    filterbank = np.zeros((mel_bin_count, linear_bin_count))
    for mel_bin in range(mel_bin_count):
        center_freq_linear_bin = linear_bin_indices[mel_bin]
        # Create a triangular filter having the current center freq.
        # The filter will start with 0 response at left_bin (if it exists)
        # and ramp up to 1.0 at center_freq_linear_bin, and then ramp
        # back down to 0 response at right_bin (if it exists).

        # Create the left side of the triangular filter that ramps up
        # from 0 to a response of 1 at the center frequency.
        if center_freq_linear_bin > 1:
            # It is possible to create the left triangular filter.
            if mel_bin == 0:
                # Since this is the first center frequency, the left side
                # must start ramping up from linear bin 0 or 1 mel bin before the center freq.
                left_bin = max(0, fft_bin_start)
            else:
                # Start ramping up from the previous center frequency bin.
                left_bin = linear_bin_indices[mel_bin - 1]
            for f_bin in range(left_bin, center_freq_linear_bin+1):
                if (center_freq_linear_bin - left_bin) > 0:
                    response = float(f_bin - left_bin)/float(center_freq_linear_bin - left_bin)
                    filterbank[mel_bin, f_bin] = response
        # Create the right side of the triangular filter that ramps down
        # from 1 to 0.
        if center_freq_linear_bin < linear_bin_count-2:
            # It is possible to create the right triangular filter.
            if mel_bin == mel_bin_count - 1:
                # Since this is the last mel bin, we must ramp down to response of 0
                # at the last linear freq bin.
                right_bin = min(linear_bin_count - 1, fft_bin_stop)
            else:
                right_bin = linear_bin_indices[mel_bin + 1]
            for f_bin in range(center_freq_linear_bin, right_bin+1):
                if (right_bin - center_freq_linear_bin) > 0:
                    response = float(right_bin - f_bin)/float(right_bin - center_freq_linear_bin)
                    filterbank[mel_bin, f_bin] = response
        filterbank[mel_bin, center_freq_linear_bin] = 1.0

    return  filterbank


def stft_for_reconstruction(x, fft_size, hopsamp):
    """Compute and return the STFT of the supplied time domain signal x.

    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):

    Returns:
        The STFT. The rows are the time slices and columns are the frequency bins.
    """
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array([np.fft.rfft(window*x[i:i+fft_size])
                     for i in range(0, len(x)-fft_size, hopsamp)])


def istft_for_reconstruction(X, fft_size, hopsamp):
    """Invert a STFT into a time domain signal.

    Args:
        X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
        fft_size (int):
        hopsamp (int): The hop size, in samples.

    Returns:
        The inverse STFT.
    """
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n,i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x


def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size, hopsamp, iterations):
    """Reconstruct an audio signal from a magnitude spectrogram.

    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.

    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
            and the columns correspond to frequency bins.
        fft_size (int): The FFT size, which should be a power of 2.
        hopsamp (int): The hope size in samples.
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
            is sufficient.

    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    time_slices = magnitude_spectrogram.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct, fft_size, hopsamp)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
        diff = np.sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff), end='\r')
    return x_reconstruct
