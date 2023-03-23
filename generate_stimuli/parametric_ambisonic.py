from librosa import stft, istft
import numpy as np
import spaudiopy
import scipy.signal as signal
import soundfile


def compose_ambisonic_signal(source_sigs, souce_dirs, diffuse_ambi_signal,
                             order):
    '''
    Puts together an ambisonic signal

    source_sigs: ndarray (SxL)
    source_dirs: ndarray (Sx2)
    diffuse_ambi_signal: diffuse part of the sig, ndarray ((order+1)^2, L)
    order: ambisonics order: int

    returns: ambisonics signal: ndarray ((order+1)^2, L)
    '''

    sh_mat = spaudiopy.sph.sh_matrix(order, souce_dirs[0, :], souce_dirs[1, :],
                                     'real')

    return (source_sigs[:, None, :] @ sh_mat)[:, 0, :] + diffuse_ambi_signal


def compose_parametric_ambisonic_signal(source_sigs, source_dirs, is_distinct,
                                        diffuse_ambi_signal, order,
                                        winlen_seconds, fs,
                                        num_directional_components,
                                        ambient_type,
                                        decorrelation_filter_path):
    '''
    Puts together an ambisonic signal from an oracle parametric representation
    using the <num_directional_components> loudest directional signals that are
    not masked by preceding direct components in each time-frequency bin

    source_sigs: ndarray (SxL)
    source_dirs: ndarray (Sx2)
    is_distinct: 1d-array (S)
    diffuse_ambi_signal: diffuse part of the sig, ndarray ((order+1)^2, L)
    order: ambisonics order: int
    winlen_seconds: the window length in seconds (STFT uses 50% overlap and 
        sqrt(Hann) windows)
    fs: sampling frequency
    num_directional_components: how many directional components in each 
        time-frequency bin: int
    ambient_type: how to represent unselected (image) sources: str, either
        'isotropic' or 'foa'
    decorrelation_filter_path: path to decorrelation filter wav file: str


    returns: ambisonics signal: ndarray ((order+1)^2, L)
    '''
    azi = source_dirs[0, :]
    zen = source_dirs[1, :]

    # sh matrix for all (image) sources
    sh_mat = spaudiopy.sph.sh_matrix(order, azi, zen, 'real')

    # omni-channel of diffuse stream to be passed through a decorrelator
    w_diffuse_stream = diffuse_ambi_signal[:, 0]

    decorr_filter_td, fs_filter = soundfile.read(decorrelation_filter_path)
    
    assert fs_filter == fs

    # prepare stft
    winlen = int(winlen_seconds / 2 * fs) * 2
    hopsize = int(winlen_seconds / 2 * fs)
    window = np.sqrt(signal.windows.hann(winlen, sym=False))
    siglen = source_sigs.shape[0]

    def my_stft(x):
        return stft(x,
                    n_fft=winlen,
                    hop_length=hopsize,
                    win_length=winlen,
                    window=window)

    def my_istft(X):
        return istft(X,
                     hop_length=hopsize,
                     win_length=winlen,
                     n_fft=winlen,
                     window=window,
                     length=siglen)

    source_specs = my_stft(source_sigs.T)
    source_mag = np.abs(source_specs)


    sources_order = np.argsort(source_mag * is_distinct[:, None, None], axis=0)[::-1,
                                                    ...]  #descending order
                                                    
    sources_used_ind = sources_order[:num_directional_components, ...]
    sources_unused_ind = sources_order[num_directional_components:, ...]

    # take the loudest (image) sources as well as their SH coefficients ...
    used_sources = np.take_along_axis(source_specs, sources_used_ind, axis=0)
    used_sources_sh_mat = np.take_along_axis(sh_mat[..., None, None],
                                             np.stack((order + 1)**2 *
                                                      [sources_used_ind], 1),
                                             axis=0)
    used_sources = used_sources.transpose(1, 2, 0)[..., None, :]
    used_sources_sh_mat = used_sources_sh_mat.transpose(
        2, 3, 0, 1)  # reshape for matmul
    
    # ... to form a hoa directional part
    hoa_directional_part = (used_sources @ used_sources_sh_mat)[..., 0, :]
    hoa_directional_part = hoa_directional_part.transpose(2, 0, 1)
    hoa_directional_part_td = my_istft(hoa_directional_part).T

    # take all other (image) sources as well as their SH coefficients ...
    unused_sources = np.take_along_axis(source_specs,
                                        sources_unused_ind,
                                        axis=0)
    unused_sources_sh_mat = np.take_along_axis(
        sh_mat[..., None, None],
        np.stack((order + 1)**2 * [sources_unused_ind], 1),
        axis=0)

    # ..to create an ambient stream from the unused (image) sources
    if unused_sources.shape[0] > 0:
        unused_sum_diff = np.sum(unused_sources, axis=0)
        if ambient_type == 'foa':
            Yamb = unused_sources_sh_mat[:, :4, :, :]
            Yamb = Yamb.transpose(2, 3, 0, 1)
            ambient_part = (
                unused_sources.transpose(1, 2, 0)[..., None, :] @ Yamb)[...,
                                                                        0, :]
            ambient_part = ambient_part.transpose(2, 0, 1)
            amb_unused_td = my_istft(ambient_part).T

    # decorrelate diffuse component
    diff_sh_td = signal.oaconvolve(decorr_filter_td,
                                   w_diffuse_stream[:, None],
                                   axes=(0, ))[:w_diffuse_stream.shape[0], :]

    # put signals together
    if unused_sources.shape[0] > 0:
        decorr_unused_td = my_istft(unused_sum_diff[None, ...])
        # decorrelate ambient stream
        unused_decorr_sh_td = 1 / np.sqrt(4 * np.pi) * \
            signal.oaconvolve(decorr_filter_td,
            decorr_unused_td.T, axes=(0,))[:decorr_unused_td.shape[1], :]
        if ambient_type == 'foa':
            # override first order channels
            unused_decorr_sh_td[:, :4] = amb_unused_td[:decorr_unused_td.
                                                       shape[1], :]
            diff_sh_td[:, :4] = diffuse_ambi_signal[:, :4]                                       
        return hoa_directional_part_td + diff_sh_td + unused_decorr_sh_td
    else:
        return hoa_directional_part_td + diff_sh_td