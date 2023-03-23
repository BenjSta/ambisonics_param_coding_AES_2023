import librosa
import sounddevice
import spaudiopy
import numpy as np
import soundfile
import scipy.signal as signal
from generate_stimuli import binaural_decode

FS = 44100
PLAYBACK_DECORR_NOISE = False
WRITE_FILTERS = False
ORDER = 3
DECODER_FILTER_PATH = 'mag_ls_binaural_decoder_weights/irsOrd3.wav'
DECORR_LEN = int(0.06 * FS / 2) * 2
EQ_LEN = int(0.06 * FS / 2) * 2


### create decorrelation noise sequences
# create noise at 24 virtual loudspeaker positions
tgrid = spaudiopy.grids.load_t_design(6) 
w = 10**(-1 * np.arange(DECORR_LEN) /
         DECORR_LEN)[:, None]  # 20dB decay over noise length
decorrelator = w * np.random.normal(0, 1, (DECORR_LEN, tgrid.shape[0]))
azi, zen, r = spaudiopy.utils.cart2sph(tgrid[:, 0], tgrid[:, 1], tgrid[:, 2])
gridmat = spaudiopy.sph.sh_matrix(ORDER, azi, zen, 'real')
decorrelator = decorrelator @ gridmat  # encode into sh domain
decorrelator = decorrelator / np.sqrt(
    np.sum(decorrelator**2, axis=0,
           keepdims=True))  # normalize (unit gain per channel)

test_noise = np.random.randn(10 * FS, (ORDER + 1)**2)  # create white noise
test_noise_w = test_noise[:, 0]
test_noise_decorr = signal.fftconvolve(test_noise_w[:, None],
                                       decorrelator,
                                       axes=0)
test_noise_binaural = binaural_decode(test_noise, FS, DECODER_FILTER_PATH)
test_noise_decorr_binaural = binaural_decode(test_noise_decorr, FS,
                                             DECODER_FILTER_PATH)


window = signal.windows.hann(EQ_LEN, sym=False)
stft = librosa.stft(test_noise_binaural.T,
                    n_fft=EQ_LEN,
                    hop_length=EQ_LEN // 2,
                    win_length=EQ_LEN,
                    window=window)
welch_periodogram_ref = np.mean(np.linalg.norm(stft, axis=-1)**2, axis=0)
stft = librosa.stft(test_noise_decorr_binaural.T,
                    n_fft=EQ_LEN,
                    hop_length=EQ_LEN // 2,
                    win_length=EQ_LEN,
                    window=window)
welch_periodogram_decorr = np.mean(np.linalg.norm(stft, axis=-1)**2, axis=0)

### minimum phase filter using cepstrum
eq = np.sqrt(welch_periodogram_ref / welch_periodogram_decorr)
c = np.fft.irfft(np.log(eq), axis=0)
cm = np.concatenate([
    c[0, None], c[1:EQ_LEN // 2] + c[-1:EQ_LEN // 2:-1], c[EQ_LEN // 2, None],
    np.zeros(EQ_LEN // 2 - 1)
])
cf = np.exp(np.fft.fft(cm))
eq_td = np.real(np.fft.ifft(cf))
decorr_complete = signal.fftconvolve(eq_td[:, None], decorrelator,
                                     axes=0)  # equalized decorrelator
decorr_complete *= 1/np.sqrt(np.sum(decorr_complete**2, axis=0, keepdims=True))

### create new test noise (just to make sure the method is not dependent on the signal)
test_noise = 0.01 * np.random.randn(3 * FS, (ORDER + 1)**2)
test_noise_w = test_noise[:, 0]
test_noise_decorr_eq = signal.fftconvolve(test_noise_w[:, None],
                                          decorr_complete,
                                          axes=0)
test_noise_decorr_no_eq = signal.fftconvolve(test_noise_w[:, None],
                                             decorrelator,
                                             axes=0)

test_noise_binaural = binaural_decode(test_noise, FS, DECODER_FILTER_PATH)
test_noise_decorr_binaural_eq = binaural_decode(test_noise_decorr_eq, FS,
                                                DECODER_FILTER_PATH)
test_noise_decorr_binaural_no_eq = binaural_decode(test_noise_decorr_no_eq, FS,
                                                   DECODER_FILTER_PATH)

# play back the filtered noise
if PLAYBACK_DECORR_NOISE:
    sounddevice.play(test_noise_binaural, FS, blocking=True)
    sounddevice.play(test_noise_decorr_binaural_eq, FS, blocking=True)
    sounddevice.play(test_noise_decorr_binaural_no_eq, FS, blocking=True)

# write to file
if WRITE_FILTERS:
    soundfile.write('decorr_filter_with_eq.wav',
                   decorr_complete,
                   FS,
                   subtype='PCM_24')
