#all parameters defined

class Preprocessing:
    BATCH_SIZE = 8
    MAX_TIME = 4
    NCEPS = 40
    NMELS = 40
    PREEMPCOEFF = 0.97
    FS = 16000
    WINDOW = 'hann'
    WINLEN = 512
    WINSHIFT = WINLEN//2
    NFFT = 512
    DETREND = False
    ONESIDED = True
    BOUNDARY = None
    PADDED = False
    eps = 0.001

class Postprocessing:
    DENOISE_FACTOR = 5  # the larger it is, the smoother curve will be
    GRIFFIN_LIM_ITE = 277
