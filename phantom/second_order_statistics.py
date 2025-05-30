import numpy as np


def autocorrelation(times, taud, lam):
    """
    Returns the normalized autocorrelation of a shot noise process.
    Input:
        times:  ndarray, float. Time lag.
        taud: float, pulse duration time.
        lam:  float, pulse asymmetry parameter. Related to pulse rise time by tr = l * td and pulse fall time by tf = (1-l) * tf.
    Output:
        ndarray, float. Autocorrelation at time lag tau.
    """
    assert taud > 0
    assert lam >= 0
    assert lam <= 1

    eps = 1e-8

    if np.abs(lam) < eps or np.abs(lam - 1) < eps:
        return np.exp(-np.abs(times) / taud)

    if np.abs(lam - 0.5) < eps:
        return (1 + 2 * np.abs(times) / taud) * np.exp(-2 * np.abs(times) / taud)

    exp1 = (1 - lam) * np.exp(-np.abs(times) / (taud * (1 - lam)))
    exp2 = lam * np.exp(-np.abs(times) / (taud * lam))
    return (exp1 - exp2) / (1 - 2 * lam)


def power_spectral_density(omega, taud, lam):
    """
    Returns the power spectral density of a shot noise process,
    given by
    PSD(omega) = 2.0 * taud / [(1 + (1 - l)^2 omega^2 taud^2) (1 + l^2 omega^2 taud^2)]
    The power spectral density is normalized such that :math:`\int_0^\inf S(\omega) d\omega = 2\pi`, which adds a factor two to the above equation.
    Input:
        omega...: ndarray, float: Angular frequency
        taud......: float, pulse duration time
        lam.......: float, pulse asymmetry parameter.
    Output:
        psd.....: ndarray, float: Power spectral density
    """
    assert taud > 0
    assert lam >= 0
    assert lam <= 1

    if lam in [0, 1]:
        return 4 * taud / (1 + (taud * omega) * (taud * omega))
    elif lam == 0.5:
        return 64 * taud / (4 + (taud * omega) * (taud * omega)) ** 2

    f1 = 1 + ((1 - lam) * taud * omega) * (1.0 - lam) * taud * omega
    f2 = 1 + (lam * taud * omega) * (lam * taud * omega)
    return 4 * taud / (f1 * f2)
