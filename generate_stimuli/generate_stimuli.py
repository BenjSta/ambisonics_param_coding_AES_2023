import numpy as np
import soundfile
from utils.noise_psd import pink_noise
import parametric_ambisonic as pamb
import scipy.signal as signal
from utils.room_simulation import simulate_simple_room
from spaudiopy.utils import cart2sph
from utils.vst_wrappers import HarpexUpmixerWrapper

def binaural_decode(x, fs, hrir_path):
    hrir, fs_binaural = soundfile.read(hrir_path)
    assert fs == fs_binaural

    # compute multiplier for left/right multiplier
    nm = np.arange(0, hrir.shape[1])
    n = np.floor(np.sqrt(nm))
    m = nm - n**2 - n
    mult = np.zeros(hrir.shape[1])
    mult[m >= 0] = 1
    mult[m < 0] = -1

    left_sh = signal.fftconvolve(x, hrir, axes=[
        0,
    ])
    left_sh = left_sh[:x.shape[0], :]
    right = np.sum(left_sh * mult[None, :], axis=-1)
    left = np.sum(left_sh, axis=-1)

    return np.stack([left, right], axis=-1)

def generate_trial(source_signals,
                   out_path_name_prefix,
                   directions,
                   room_dim,
                   rt60,
                   drr_db,
                   receiver_pos,
                   normalise_db,
                   fs,
                   order,
                   harpex=None):
    '''
    generate a trial with binaural and ambisonic stimuli

    source_signals: numpy array (SxL)
    target_directory: where to save
    directions: numpy array (Sx2)
    room_dim: None or numpy array (3,)
    rt60: reverberation time: float (ignored if room_dim is None)
    drr_db: direct-to-reverberant ratio: float (ignored if room_dim is None)
    receiver_pos: receiver position in the room numpy array (3,) 
        (ignored if room_dim is None)
    normalise_db: target peak value for omnidirectional channel float
    fs: sampling rate: float
    order: ambisonics order: int
    '''
    if room_dim is not None:
        # room simulation
        C = 340
        MAXDELAY_DISCRETE = np.max(room_dim) * (ROOM_SIM_MIN_IMAGE_ORDER +
                                                1) / C
        maxdelay_discrete_samples = int(MAXDELAY_DISCRETE * fs)
        directions, delays, amplitudes, is_distinct, diffuse_envelope = \
            simulate_simple_room(rt60, drr_db, room_dim, receiver_pos,
            directions, MAXDELAY_DISCRETE, fs, C)

        all_discrete_signals = []
        all_is_distinct = []
        all_discrete_directions = []
        diffuse_sh = np.zeros((source_signals.shape[0], (order + 1)**2))

        # iterate over all sources
        for i in range(source_signals.shape[1]):
            for j in range(directions[i].shape[0]):
                d = delays[i][j]
                s = amplitudes[i][j] * np.pad(source_signals[:, i],
                                              ((d, 0), ))[:-d]
                all_is_distinct.append(is_distinct[i][j])
                all_discrete_signals.append(s)
                all_discrete_directions.append(directions[i][j, :])

            diffuse_ir = np.random.randn(diffuse_envelope.shape[0],
                                         (order + 1)**2)
            diffuse_ir = diffuse_ir * diffuse_envelope[:, None]
            diffuse_sh += 1 / np.sqrt(4 * np.pi) * np.pad(
                signal.oaconvolve(
                    source_signals[:, i, None], diffuse_ir, axes=0),
                ((maxdelay_discrete_samples, 0),
                 (0, 0)))[:source_signals.shape[0], :]

        # stack all direct sounds and reflections of all sources together
        all_discrete_directions = np.stack(all_discrete_directions, axis=-1)
        all_discrete_directions_azi, all_discrete_directions_zen, _ = \
            cart2sph(all_discrete_directions[0, :],
            all_discrete_directions[1, :], all_discrete_directions[2, :])
        all_discrete_directions = np.stack(
            [all_discrete_directions_azi, all_discrete_directions_zen], axis=0)

        all_discrete_signals = np.stack(all_discrete_signals, axis=-1)
        all_is_distinct = np.stack(all_is_distinct, axis=-1)

    else:
        all_discrete_signals = source_signals
        all_is_distinct = np.stack([True] * all_discrete_signals.shape[1])
        all_discrete_directions = directions
        diffuse_sh = np.zeros((source_signals.shape[0], (order + 1)**2))

    hoa = pamb.compose_ambisonic_signal(all_discrete_signals,
                                        all_discrete_directions, diffuse_sh,
                                        order)

    w_max = np.max(np.abs(hoa[:, 0]))
    normalization = 10**(normalise_db / 20) / w_max
    hoa *= normalization

    soundfile.write(out_path_name_prefix + '_ref_amb.wav',
                    hoa,
                    fs,
                    subtype='PCM_24')
    hoa_binaural = binaural_decode(hoa, fs, HRIR_PATH)
    assert np.max(np.abs(hoa_binaural)) <= 1
    soundfile.write(out_path_name_prefix + '_ref_binau.wav',
                    hoa_binaural,
                    fs,
                    subtype='PCM_24')

    foa = hoa[:, :4].copy()
    foa_binaural = binaural_decode(foa * 10**(-1.5 / 20), fs, HRIR_PATH_FOA)
    assert np.max(np.abs(foa_binaural)) <= 1
    soundfile.write(out_path_name_prefix + '_foa_binau.wav',
                    foa_binaural,
                    fs,
                    subtype='PCM_24')
    soundfile.write(out_path_name_prefix + '_foa_amb.wav',
                    foa,
                    fs,
                    subtype='PCM_24')

    hoa_param1 = pamb.compose_parametric_ambisonic_signal(
        all_discrete_signals, all_discrete_directions, all_is_distinct,
        diffuse_sh, order, PARAM_WINLEN, fs, 1, 'isotropic',
        'decorr_filter_with_eq.wav')
    hoa_param1 *= normalization
    soundfile.write(out_path_name_prefix + '_param1_amb.wav',
                    hoa_param1,
                    fs,
                    subtype='PCM_24')
    hoa_param1_binaural = binaural_decode(hoa_param1, fs, HRIR_PATH)
    assert np.max(np.abs(hoa_param1_binaural)) <= 1
    soundfile.write(out_path_name_prefix + '_param1_binau.wav',
                    hoa_param1_binaural,
                    fs,
                    subtype='PCM_24')

    hoa_dir_param1 = pamb.compose_parametric_ambisonic_signal(
        all_discrete_signals, all_discrete_directions, all_is_distinct,
        diffuse_sh, order, PARAM_WINLEN, fs, 1, 'foa',
        'decorr_filter_with_eq.wav')
    hoa_dir_param1 *= normalization
    soundfile.write(out_path_name_prefix + '_foa_amb_param1_amb.wav',
                    hoa_dir_param1,
                    fs,
                    subtype='PCM_24')
    hoa_dir_param1_binaural = binaural_decode(hoa_dir_param1, fs, HRIR_PATH)
    assert np.max(np.abs(hoa_dir_param1_binaural)) <= 1
    soundfile.write(out_path_name_prefix + '_foa_amb_param1_binau.wav',
                    hoa_dir_param1_binaural,
                    fs,
                    subtype='PCM_24')

    hoa_param2 = pamb.compose_parametric_ambisonic_signal(
        all_discrete_signals, all_discrete_directions, all_is_distinct,
        diffuse_sh, order, PARAM_WINLEN, fs, 2, 'isotropic',
        'decorr_filter_with_eq.wav')
    hoa_param2 *= normalization
    soundfile.write(out_path_name_prefix + '_param2_amb.wav',
                    hoa_param2,
                    fs,
                    subtype='PCM_24')
    hoa_param2_binaural = binaural_decode(hoa_param2, fs, HRIR_PATH)
    assert np.max(np.abs(hoa_param2_binaural)) <= 1
    soundfile.write(out_path_name_prefix + '_param2_binau.wav',
                    hoa_param2_binaural,
                    fs,
                    subtype='PCM_24')

    hoa_dir_param2 = pamb.compose_parametric_ambisonic_signal(
        all_discrete_signals, all_discrete_directions, all_is_distinct,
        diffuse_sh, order, PARAM_WINLEN, fs, 2, 'foa',
        'decorr_filter_with_eq.wav')
    hoa_dir_param2 *= normalization
    soundfile.write(out_path_name_prefix + '_foa_amb_param2_amb.wav',
                    hoa_dir_param2,
                    fs,
                    subtype='PCM_24')
    hoa_dir_param2_binaural = binaural_decode(hoa_dir_param2, fs, HRIR_PATH)
    assert np.max(np.abs(hoa_dir_param2_binaural)) <= 1
    soundfile.write(out_path_name_prefix + '_foa_amb_param2_binau.wav',
                    hoa_dir_param2_binaural,
                    fs,
                    subtype='PCM_24')

    s = 10**(6 / 20) * np.stack(2 * [hoa[:, 0]], -1)
    assert np.max(np.abs(s)) <= 1
    soundfile.write(out_path_name_prefix + '_mono.wav',
                    s,
                    fs,
                    subtype='PCM_24')

    if harpex is not None:
        hoa_harpex = harpex.process(foa * 10**((1.5 / 20)))
        soundfile.write(out_path_name_prefix + '_harpex_amb.wav',
                        hoa_harpex,
                        fs,
                        subtype='PCM_24')
        hoa_harpex_binaural = binaural_decode(hoa_harpex, fs, HRIR_PATH)
        soundfile.write(out_path_name_prefix + '_harpex_binau.wav',
                        hoa_harpex_binaural,
                        fs,
                        subtype='PCM_24')


if __name__ == '__main__':
    FS = 44100
    PARAM_WINLEN = 0.02
    HRIR_PATH_FOA = 'mag_ls_binaural_decoder_weights/irsOrd1.wav'
    HRIR_PATH = 'mag_ls_binaural_decoder_weights/irsOrd3.wav'
    ROOM_SIM_MIN_IMAGE_ORDER = 2
    AMB_ORDER = 11

    harpex = HarpexUpmixerWrapper(FS, 'vst_plugins/Harpex-X.dll')
    #harpex = None

    main_room = np.array([7, 6, 3.5])
    listener_pos = np.array([3.5, 4, 1.5])

    for reverb in ['anech', 'medrev', 'strongrev']:

        if reverb == 'anech':
            room = None
            drr = np.nan
            t60 = np.nan
        else:
            room = main_room
            if reverb == 'medrev':
                drr = 0
                t60 = 0.3
            elif reverb == 'strongrev':
                drr = -6
                t60 = 0.6

        for scenario in [
                'pink_noise', 'drums+saw', 'string_quartet', 'two_speakers',
                'speech+noise'
        ]:
            name = scenario + '_' + reverb
            normalise_db = -12
            if scenario == 'pink_noise':
                x1 = pink_noise(int(2.5 * FS))
                x2 = pink_noise(int(2.5 * FS))
                signals = np.stack([x1, x2], -1)
                directions = np.pi / 180 * np.array([[-45, 90], [45, 90]]).T
                normalise_db = -18

            elif scenario == 'drums+saw':
                drumbeat, _ = soundfile.read(
                    'source_audio/freesound/385676__scydan__trip-acoustic-beat_cut.wav'
                )
                sawtooth, _ = soundfile.read('source_audio/saw.wav')
                sawtooth = 0.3 * sawtooth[:int(2.5 * FS)]
                signals = np.stack([drumbeat, sawtooth], -1)
                directions = np.pi / 180 * np.array([[-90, 90], [45, 45]]).T

            elif scenario == 'string_quartet':
                violin1, _ = soundfile.read(
                    'source_audio/gomes_string_quartet/Mov1_Violin1_Haydn_StringQuartet_op76_n1_cut.wav'
                )
                violin2, _ = soundfile.read(
                    'source_audio/gomes_string_quartet/Mov1_Violin2_Haydn_StringQuartet_op76_n1_cut.wav'
                )
                viola, _ = soundfile.read(
                    'source_audio/gomes_string_quartet/Mov1_Viola_Haydn_StringQuartet_op76_n1_cut.wav'
                )
                cello, _ = soundfile.read(
                    'source_audio/gomes_string_quartet/Mov1_Cello_Haydn_StringQuartet_op76_n1_cut.wav'
                )
                signals = np.stack([violin1, violin2, viola, cello], -1)
                directions = np.pi / 180 * np.array([[-90, 90], [-30, 90],
                                                     [30, 90], [90, 90]]).T

            elif scenario == 'two_speakers':
                s1, _ = soundfile.read('source_audio/ebu_sqam/50_cut.wav')
                s2, _ = soundfile.read('source_audio/ebu_sqam/49_cut.wav')
                signals = np.stack([s1, s2], -1)
                directions = np.pi / 180 * np.array([[-45, 70], [45, 90]]).T

            elif scenario == 'speech+noise':
                s1, _ = soundfile.read('source_audio/ebu_sqam/52_cut.wav')
                fan, _ = soundfile.read(
                    'source_audio/freesound/594984__hdomst__fan-noise_cut.wav')
                s2, _ = soundfile.read('source_audio/ebu_sqam/53_cut.wav')
                keyboard, _ = soundfile.read(
                    'source_audio/freesound/489423__samsterbirdies__typing-on-a-keyboard_cut.wav'
                )
                signals = np.stack([s1, fan, s2, keyboard], -1)
                directions = np.pi / 180 * np.array([[-135, 90], [-45, 120],
                                                     [45, 70], [135, 100]]).T

            generate_trial(signals, 'rendered_audio_o11/' + name, directions, room,
                           t60, drr, listener_pos, normalise_db, FS, AMB_ORDER,
                           harpex)