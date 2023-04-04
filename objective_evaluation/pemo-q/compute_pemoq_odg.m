function odg = compute_pemoq_odg(ref,test,fs_wav,modproc)
    [mref_l, fs] = pemo_internal(ref(:, 1), fs_wav, modproc);
    [mtest_l, ~] = pemo_internal(test(:, 1), fs_wav, modproc);
    psm_t_l = pemo_metric(mref_l, mtest_l, fs);
    [mref_r, fs] = pemo_internal(ref(:, 2), fs_wav, modproc);
    [mtest_r, ~] = pemo_internal(test(:, 2), fs_wav, modproc);
    psm_t_r = pemo_metric(mref_r, mtest_r, fs);
    psm_t = min(psm_t_l, psm_t_r);
    % ODG mapping as in Huber & Kollmeier: "PEMO-Q—A new method for 
    % objective audio quality assessment using a model of auditory 
    % perception", 2006.
    if psm_t < 0.864
        odg = max(-0.22 / (psm_t - 0.98) - 4.13, -4);
    else
        odg = 16.4 * psm_t - 16.4;
    end
end