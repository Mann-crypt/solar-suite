"""
Aeromal — ported from the Streamlit "Aeromal" page.
Two modes: Curtailment (peak-shaping with a semicircular cap) and
No-Curtailment (straight 95th-percentile + symmetry-shift smoothing).

Password gate is enforced server-side in the FastAPI route, not here —
never trust a client-side-only check.
"""

import numpy as np
from scipy.signal import savgol_filter

AEROMAL_PASSWORD = "asdfghjkl;'"  # TODO: move to an environment variable before shipping


def check_password(password: str) -> bool:
    return password == AEROMAL_PASSWORD


def _best_symmetry_shift(profile, reference):
    least_error = np.inf
    best_shift = 0
    for i in range(96):
        sh = np.roll(profile, -i)
        sym = (profile + sh[::-1]) / 2
        error = np.sqrt(np.mean((reference - sym) ** 2))
        if error < least_error:
            least_error = error
            best_shift = i
    return best_shift


def solar_cap_curve(y, peak_cap, target_width, window_length, power_availability):
    y = np.array(y, dtype=float)
    n = len(y)
    para = np.zeros(n)

    ys = savgol_filter(y, 7, 2)
    grad = np.gradient(ys)

    left_peak = np.argmax(ys[: n // 2])
    left_start = np.argmax(grad[:left_peak])
    x_left = np.arange(left_start, left_peak)
    y_left = ys[left_start:left_peak]
    m1, c1 = np.polyfit(x_left, y_left, 1)

    right_peak = np.argmax(ys[n // 2:]) + n // 2
    threshold = 0.02 * np.max(ys)
    active_idx = np.where(ys > threshold)[0]
    right_end = active_idx[-1]
    x_right = np.arange(right_peak, right_end)
    y_right = ys[right_peak:right_end]
    m2, c2 = np.polyfit(x_right, y_right, 1)

    A = (1 / m2) - (1 / m1)
    B = (c1 / m1) - (c2 / m2)
    trip = (target_width - B) / A

    peak_left_idx = int(round((trip - c1) / m1))
    peak_right_idx = int(round((trip - c2) / m2))
    trip = max(0, trip)

    if peak_cap >= trip:
        for i in range(n):
            val = m1 * i + c1
            para[i] = min(val, trip)
            if val >= trip:
                peak_left_idx = i
                break

        right_curve = np.zeros(n)
        for i in range(n - 1, -1, -1):
            val = m2 * i + c2
            right_curve[i] = min(val, trip)
            if val >= trip:
                peak_right_idx = i
                break

        para = np.maximum(para, right_curve)

        width = peak_right_idx - peak_left_idx
        dome_height = max(20, 0.12 * trip)
        x = np.linspace(-1, 1, width)
        shape = np.sqrt(np.maximum(0, 1 - x ** 2))
        dome = trip + dome_height * shape
        dome[0] = trip
        dome[-1] = trip
        para[peak_left_idx:peak_right_idx] = dome

        para = savgol_filter(para, window_length, 3)
        para = np.clip(para, 0, None)

        para[:left_start] = ys[:left_start]
        para[right_end:] = ys[right_end:]

        para = savgol_filter(para, 7, 3) * power_availability / 100
        para = np.clip(para, 0, None)
        para = np.where(para < 0.2, 0, para)
        return para

    else:
        for i in range(n):
            val = m1 * i + c1
            para[i] = val
            if val >= peak_cap:
                peak_left_idx = i
                break

        right_curve = np.zeros(n)
        for i in range(n - 1, -1, -1):
            val = m2 * i + c2
            right_curve[i] = val
            if val >= peak_cap:
                peak_right_idx = i
                break

        para = np.maximum(para, right_curve)
        para[peak_left_idx:peak_right_idx] = peak_cap

        para = np.clip(para, 0, peak_cap)
        para = savgol_filter(para, window_length, 3)

        para[:left_start] = ys[:left_start]
        para[right_end:] = ys[right_end:]
        para = savgol_filter(para, 7, 3) * power_availability / 100
        para = np.clip(para, 0, peak_cap)
        para = np.where(para < 1, 0, para)
        return para


def run_curtailment(power, peak_cap, target_width, window_length, power_availability, shift=None) -> dict:
    power = np.asarray(power, dtype=float)
    if len(power) == 0 or len(power) % 96 != 0:
        raise ValueError("Number of rows must be non-zero and divisible by 96.")
    if not np.any(power > 0):
        raise ValueError("Power values must contain at least one positive value.")

    days = len(power) // 96
    a = power.reshape(days, 96)
    ap = np.percentile(a, 95, axis=0) * 1.03

    final_smooth = solar_cap_curve(ap, peak_cap, target_width, window_length, power_availability)

    if shift is None:
        shift = _best_symmetry_shift(final_smooth, final_smooth)

    sh = np.roll(final_smooth, -int(shift))
    final_smooth_sym = (final_smooth + sh[::-1]) / 2

    return {
        "mode": "curtailment",
        "shift": int(shift),
        "chart": {
            "x": list(range(96)),
            "generation": np.round(ap, 4).tolist(),
            "profile": np.round(final_smooth, 4).tolist(),
            "sym_profile": np.round(final_smooth_sym, 4).tolist(),
        },
    }


def run_no_curtailment(power, window_length=11, power_availability=100, shift=None) -> dict:
    power = np.asarray(power, dtype=float)
    if len(power) == 0 or len(power) % 96 != 0:
        raise ValueError("Number of rows must be non-zero and divisible by 96.")

    days = len(power) // 96
    a = power.reshape(days, 96)
    ap = np.percentile(a, 95, axis=0)

    s = savgol_filter(ap, window_length=window_length, polyorder=3)

    if shift is None:
        shift = _best_symmetry_shift(s, ap)

    alpha = 0.50
    sh = np.roll(s, -shift)
    sym = alpha * s + (1 - alpha) * sh[::-1]

    thr = 0.1
    idx = np.where(ap > thr)[0]
    if len(idx) > 0:
        start, end = idx[0], idx[-1]
        blend = 8

        w = np.linspace(1, 0, blend)
        sym[start + 1:start + 1 + blend] = (
            w * ap[start + 1:start + 1 + blend] + (1 - w) * sym[start + 1:start + 1 + blend]
        )
        w = np.linspace(0, 1, blend)
        sym[end - blend:end] = w * ap[end - blend:end] + (1 - w) * sym[end - blend:end]

    s = savgol_filter(ap, window_length=11, polyorder=3)
    sym = savgol_filter(sym, window_length=11, polyorder=3)

    s = np.clip(s, 0, None)
    sym = np.clip(sym, 0, None)
    s = np.where(s < 0.1, 0, s)
    sym = np.where(sym < 0.1, 0, sym)

    s = s * power_availability / 100
    sym = sym * power_availability / 100

    return {
        "mode": "no_curtailment",
        "shift": int(shift),
        "chart": {
            "x": list(range(96)),
            "percentile_95": np.round(ap, 4).tolist(),
            "profile": np.round(s, 4).tolist(),
            "sym_profile": np.round(sym, 4).tolist(),
        },
    }
