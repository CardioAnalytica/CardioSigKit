import copy
import numpy as np
from tqdm import tqdm
import neurokit2 as nk
from scipy.signal import medfilt, savgol_filter
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
import scipy
from numpy.linalg import norm
from iodeeplib.std import printf as echo
import biosppy
from scipy.interpolate import UnivariateSpline

def normalize(signals, corrupted_signals, inplace=True, feature_range=(0,1)):
    """
    Normalize the signals (raw signals and corrupted signals) for denoising autoencoder training, it normalizes pair(s) of signals (clean, corrupted) between a given interval, the noramolzation is relative to both signals to maintain relative aspect ratio.
    Parameters
    ----------
    signals: a numpy array of raw signals
    corrupted_signals: a numpy array of noisy signals (corrupted ones)
    inplace: boolean, if True update input signals else perform a deep copy of the input signals and apply normalization to it and return the results. It won't return any results if inplace=True.
    feature_range: tuple, the feature range between which signals will be normalized

    Returns: The normalized signals if inplace=True, else update inplace the signals.
    -------
    """
    assert signals.shape == corrupted_signals.shape, "Shapes of signals and corrupted signals must match"
    if not inplace:
        signals = copy.deepcopy(signals)
        corrupted_signals = copy.deepcopy(corrupted_signals)
    if len(signals.shape) == 1:
        upper = feature_range[1]
        lower = feature_range[0]
        max_value = max(np.max(signals), np.max(corrupted_signals))
        min_value = min(np.min(signals), np.min(corrupted_signals))
        m = (upper - lower) / (max_value - min_value)
        signals = (m * (signals - min_value)) + lower
        corrupted_signals = (m * (corrupted_signals - min_value)) + lower
        if not inplace:
            return signals, corrupted_signals
    elif len(signals.shape) == 2:
        for i in tqdm(range(len(signals))):
            if np.equal(np.unique(signals[i, :]), 0).all() or np.equal(np.unique(corrupted_signals[i, :]), 0).all():
                continue
            upper = feature_range[1]
            lower = feature_range[0]
            max_value = max(np.max(signals[i, :]), np.max(corrupted_signals[i, :]))
            min_value = min(np.min(signals[i, :]), np.min(corrupted_signals[i, :]))
            m = (upper - lower) / (max_value - min_value)
            signals[i, :] = (m * (signals[i, :] - min_value)) + lower
            corrupted_signals[i, :] = (m * (corrupted_signals[i, :] - min_value)) + lower
        if not inplace:
            return signals, corrupted_signals
    else:
        raise AssertionError("Incorrect shape for signals")

def standardize(signals, corrupted_signals, inplace=True, make_positive=False, lower_level=None):
    """
    Standardize the signals (raw signals and corrupted signals) for denoising autoencoder training, it standardizes pair(s) of signals (clean, corrupted) by the mean and standard deviation of the clean signal to maintain relative aspect ratio?
    Parameters
    ----------
    signals: a numpy array of raw signals
    corrupted_signals: a numpy array of noisy signals (corrupted ones)
    inplace: boolean, if True update input signals else perform a deep copy of the input signals and apply standardization to it and return the results. It won't return any results if inplace=True.

    Returns: The standardized signals if inplace=True, else update inplace the signals.
    -------
    """
    assert signals.shape == corrupted_signals.shape, "Shapes of signals and corrupted signals must match"
    if not inplace:
        signals = copy.deepcopy(signals)
        corrupted_signals = copy.deepcopy(corrupted_signals)

    if make_positive:
        echo("Transforming negative values to positive")
    if len(signals.shape) == 1:
        mean = np.sum(signals) / len(signals)
        std = np.std(signals)
        signals = (signals - mean) / np.std(signals)
        corrupted_signals = (corrupted_signals[:] - mean) / std
        if not inplace:
            return signals, corrupted_signals
    elif len(signals.shape) == 2:
        for i in tqdm(range(len(signals))):
            mean = np.sum(signals[i, :]) / len(signals[i, :])
            std = np.std(signals[i, :])
            signals[i, :] = (signals[i, :] - mean) / np.std(signals[i, :])
            corrupted_signals[i, :] = (corrupted_signals[i, :] - mean) / std
            if make_positive:
                lower_amplitude = max(np.abs(np.min(signals[i, :])), np.abs(np.min(corrupted_signals[i, :])))

                signals[i, :] = signals[i, :] + lower_amplitude
                corrupted_signals[i, :] = corrupted_signals[i, :] + lower_amplitude
            elif lower_level is not None:
                if (np.min(corrupted_signals[i, :]) < lower_level) or (np.min(signals[i, :]) < lower_level):
                    delta = lower_level - min(np.min(corrupted_signals[i, :]), np.min(signals[i, :]))
                    signals[i, :] = signals[i, :] + delta
                    corrupted_signals[i, :] = corrupted_signals[i, :] + delta
        if not inplace:
            return signals, corrupted_signals
    else:
        raise AssertionError("Incorrect shape for signals")

def quality(signal, return_clean=False, powerline_filter=False, baseline_wander_filter=None):
    try:
        ecg_cleaned = nk.ecg_clean(signal, sampling_rate=500)
        q = nk.ecg_quality(ecg_cleaned, sampling_rate=500, approach="fuzzy", method="averageQRS")
        if baseline_wander_filter is None:
            ecg_cleaned = signal
        else:
            ecg_cleaned = fix_baseline_wander(signal, sampling_rate=500, method=baseline_wander_filter["method"], args=baseline_wander_filter.get("params", None))
        if powerline_filter:
            ecg_cleaned = nk.signal_filter(signal=ecg_cleaned, sampling_rate=500, method="powerline", powerline=50)
        if return_clean:
            return np.mean(q, axis=0), ecg_cleaned
        else:
            return np.mean(q, axis=0)
    except:
        if return_clean:
            return 0, None
        else:
            return 0

def fix_baseline_wander(data, sampling_rate, method, args, return_baseline=False):
    if args is None:
        args = {}
    baseline = None
    corrected_signal = None
    if method == "median_filter":
        window_length = int(round(args["alpha1"]*sampling_rate))
        # delayBLR = round((WinSize-1)/2)
        if window_length % 2 == 0:
            window_length += 1
        baseline = medfilt(data, kernel_size=window_length)
        window_length = int(round(args["alpha2"]*sampling_rate))
        # delayBLR = delayBLR + round((WinSize-1)/2)
        if window_length % 2 == 0:
            window_length += 1
        baseline = medfilt(baseline, kernel_size=window_length)
        corrected_signal = data - baseline
    elif method == "local_regression":
        corrected_signal =  nk.signal_detrend(data, sampling_rate=sampling_rate, method="locreg")
        baseline = data - corrected_signal
    elif method == "baseline_arPLS":
        baseline = baseline_arPLS(savgol_filter(data, window_length=args.get("window_length", 13), polyorder=args.get("polyorder", 1)), lam=args.get("lam", 50000))
        corrected_signal = data - baseline
    elif method == "baseline_als_optimized":
        baseline = baseline_als_optimized(savgol_filter(data, window_length=args.get("window_length", 13), polyorder=args.get("polyorder", 1)), lam=args.get("lam", 50000), p=args.get("p", 0))
        corrected_signal = data - baseline
    if return_baseline:
        return corrected_signal, baseline
    else:
        return corrected_signal

def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    L = len(y)
    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    crit = 1
    count = 0
    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))
        crit = norm(w_new - w) / norm(w)
        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        count += 1
        if count > niter:
            # print('Maximum number of iterations exceeded')
            break
    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z

def baseline_als_optimized(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def detect_rpeaks(ecg, rate, ransac_window_size=5.0, lowfreq=35.0, highfreq=43.0):
    """
    ECG heart beat detection based on
    http://link.springer.com/article/10.1007/s13239-011-0065-3/fulltext.html
    with some tweaks (mainly robust estimation of the rectified signal
    cutoff threshold).
    """
    import warnings
    warnings.filterwarnings('ignore')
    ransac_window_size = int(ransac_window_size * rate)

    lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    # TODO: Could use an actual bandpass filter
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)

    # Square (=signal power) of the first difference of the signal
    decg = np.diff(ecg_band)
    decg_power = decg ** 2

    # Robust threshold and normalizator estimation
    thresholds = []
    max_powers = []
    for i in range(int(len(decg_power) / ransac_window_size)):
        sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
        d = decg_power[sample]
        thresholds.append(0.5 * np.std(d))
        max_powers.append(np.max(d))

    threshold = np.median(thresholds)
    max_power = np.median(max_powers)
    decg_power[decg_power < threshold] = 0

    decg_power /= max_power
    decg_power[decg_power > 1.0] = 1.0
    square_decg_power = decg_power ** 2

    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0

    mean_window_len = int(rate * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
    # lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)

    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 8.0)
    lp_energy_diff = np.diff(lp_energy)

    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1
    return zero_crossings

def clean(ecg, sampling_rate=500, type=1):
    if type == 1:
        filtered = biosppy.signals.ecg.ecg(ecg, sampling_rate, show=False)['filtered']
        return nk.ecg_clean(filtered, sampling_rate=sampling_rate)
    elif type == 2:
        return biosppy.signals.ecg.ecg(ecg, sampling_rate, show=False)['filtered']
    elif type == 3:
        return nk.ecg_clean(ecg, sampling_rate=sampling_rate)
    elif type == 0:
        return ecg
    else:
        raise ValueError("Unknown type")

def interpolate(x, s):
    indices = np.arange(0, len(x))
    new_indices = np.linspace(0, len(x)-1, s)
    spl = UnivariateSpline(indices, x, k=3,s=0)
    return spl(new_indices)

def filter_signal(signal, sampling_rate=10000, filter_window=20):
    """
    Filter the signal using two pass filters, local regression and windowed average.
    :param signal: Input signal vector in numpy array format
    :param sampling_rate: Sampling rate or frequency of input signal
    :param filter_window: Window for the windowed average filter
    :return: Filtered signal without baseline wander
    """
    # First pass, remove baseline wander
    filtered_signal, signal_baseline = fix_baseline_wander(signal, sampling_rate, "local_regression", {"alpha1": 0.2, "alpha2": 0.6}, True)
    # Move signal from 0 line to initial signal mean
    filtered_signal += np.mean(signal)
    # Second pass, apply windowed average filter
    num_pass = int(len(filtered_signal) / filter_window)
    reconstructed_signal = []
    for p in range(num_pass):
        reconstructed_signal.append(np.mean(filtered_signal[p * filter_window:(p * filter_window) + filter_window]))
    reconstructed_signal = np.array(interpolate(reconstructed_signal, len(filtered_signal)))
    return reconstructed_signal

def compute_prototypes(signal, rpeaks, rrs, window=20000):
    prototypes = []
    k = 0
    for i in range(0, len(rpeaks), 2):
        if i + 2 >= len(rpeaks):
            break
        rr = rpeaks[i + 1] - rpeaks[i]
        if rr in rrs[np.where(rrs > (np.abs(np.median(rrs)) + np.abs(np.median(rrs)) * 0.01))[0].tolist()].tolist():
            continue
        if rr in rrs[np.where(rrs < (np.abs(np.median(rrs)) - np.abs(np.median(rrs)) * 0.01))[0].tolist()].tolist():
            continue
        k += 1
        prototype_window = signal[rpeaks[i]:rpeaks[i + 2]]
        prototype_window = interpolate(prototype_window, window)
        prototypes.append(prototype_window)
    prototype = np.mean(prototypes, axis=0)
    return prototypes, prototype