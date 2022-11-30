import numpy as np
from mne.filter import filter_data


def apply_filtering(psg, sample_rate, **filter_kwargs) -> np.ndarray:
    """
    Applies the mne.filter.filter_data method on PSG array (ndarray, [N, C]) with
    parameters as specified by filter_kwargs.

    Example parameters for 0.3-35 Hz band-pass:
    filter_kwargs: {'l_freq': 0.3, 'h_freq': 35, 'method': 'fir'}

    Args:
        psg:                      A ndarray of shape [N, C] of PSG data
        sample_rate:              The sample rate of data in the PSG
        **filter_kwargs:          Filtering arguments passed to mne.filter.filter_data
    """
    return filter_data(
        psg.T, sample_rate, **filter_kwargs
    ).T
