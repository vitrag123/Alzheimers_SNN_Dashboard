
import numpy as np
from scipy.signal import welch

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4",
    "P3", "P4", "O1", "O2", "F7", "F8",
    "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
]

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


def extract_window_features(window, sfreq=500):
    """
    window shape: (19, 1000)
    Returns 171 features.
    """

    if window.shape != (19, 1000):
        raise ValueError(
            "Expected EEG window shape (19, 1000)"
        )

    features = []

    for channel in window:

        features.extend([
            np.mean(channel),
            np.var(channel),
            np.std(channel),
            np.sum(channel ** 2)
        ])

        freqs, psd = welch(
            channel,
            fs=sfreq,
            nperseg=500
        )

        for low, high in BANDS.values():

            mask = (
                (freqs >= low) &
                (freqs < high)
            )

            power = np.trapezoid(
                psd[mask],
                freqs[mask]
            )

            features.append(power)

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.shape != (171,):
        raise ValueError("Incorrect feature count")

    return features


def extract_recording_features(windows):
    """
    windows shape: (number_of_windows, 19, 1000)
    """

    return np.stack([
        extract_window_features(window)
        for window in windows
    ])
