"""
EAS Station - Emergency Alert System
Copyright (c) 2025-2026 EAS Station, LLC (KR8MER)

This file is part of EAS Station.

EAS Station is dual-licensed software:
- GNU Affero General Public License v3 (AGPL-3.0) for open-source use
- Commercial License for proprietary use

You should have received a copy of both licenses with this software.
For more information, see LICENSE and LICENSE-COMMERCIAL files.

IMPORTANT: This software cannot be rebranded or have attribution removed.
See NOTICE file for complete terms.

Repository: https://github.com/KR8MER/eas-station
"""

"""Tests for FMDemodulator's peak deviation / pilot / RDS injection metrics.

These are the one genuinely new piece of DSP in the "FM broadcast analyzer"
feature set (unlike stereo_pilot_strength/click_rate, which just needed
wiring an already-computed value through to the UI): peak_deviation_hz,
pilot_injection_hz and rds_injection_hz are derived from the raw
discriminator output by the inverse of the existing _audio_gain scale
factor (radians/sample -> real Hz). Correctness here means "does a signal
synthesized at a known deviation come back reading close to that value,"
not just "does the field exist and thread through" -- the field-plumbing
side of this feature already has ample precedent/coverage elsewhere
(see tests/test_signal_quality_history_api.py for the click_rate case).
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_core.radio.demod import DemodulatorConfig, FMDemodulator  # noqa: E402


SR = 250_000  # 250 kHz multiplex sample rate (typical FM broadcast IF)


def _make_demodulator(sample_rate: int = SR, enable_rbds: bool = False) -> FMDemodulator:
    cfg = DemodulatorConfig(
        modulation_type="FM",
        sample_rate=sample_rate,
        audio_sample_rate=48_000,
        stereo_enabled=True,
        enable_rbds=enable_rbds,
    )
    return FMDemodulator(cfg)


def _fm_modulate(inst_freq_hz: np.ndarray, sample_rate: int) -> np.ndarray:
    """Synthesize complex baseband IQ for a given instantaneous frequency
    (Hz) trajectory -- the exact inverse of what fm_discriminator recovers.

    phase[n] = phase[n-1] + 2*pi*f[n]/sample_rate; iq[n] = exp(j*phase[n]).
    """
    phase = np.cumsum(2.0 * np.pi * inst_freq_hz / sample_rate)
    return np.exp(1j * phase).astype(np.complex64)


def test_peak_deviation_matches_a_known_single_tone_modulation():
    """A carrier frequency-modulated by a single low-rate tone at an exact
    known peak deviation must read back close to that peak in Hz.

    peak_deviation_hz is deliberately the *unfiltered* composite discriminator
    output's peak -- for a single pure tone with no pilot/RDS/audio content
    layered on top, that peak is exactly the modulating tone's peak deviation.
    """
    demod = _make_demodulator()
    duration = 0.05
    n = int(SR * duration)
    t = np.arange(n) / SR
    peak_dev_hz = 50_000.0
    mod_rate_hz = 1_000.0  # well within Carson bandwidth at this deviation/SR
    inst_freq = peak_dev_hz * np.cos(2 * np.pi * mod_rate_hz * t)

    iq = _fm_modulate(inst_freq, SR)
    audio, status = demod.demodulate(iq)

    assert status is not None
    # A few percent tolerance for discriminator edge effects at chunk
    # boundaries (first/last sample), not a fundamental limitation.
    assert status.peak_deviation_hz == pytest.approx(peak_dev_hz, rel=0.05)


def test_peak_deviation_is_zero_for_an_unmodulated_carrier():
    """A pure, unmodulated carrier (constant instantaneous frequency = 0
    offset) must read back ~0 Hz peak deviation, not noise or a nonzero
    artifact from the discriminator's own quantization."""
    demod = _make_demodulator()
    n = int(SR * 0.02)
    iq = np.ones(n, dtype=np.complex64)  # constant phase -> zero deviation

    audio, status = demod.demodulate(iq)

    assert status is not None
    assert status.peak_deviation_hz < 100.0  # near-zero, allowing for float noise


def test_pilot_injection_hz_matches_a_known_pilot_amplitude():
    """A composite signal containing only a 19 kHz pilot at a known peak
    deviation must read back an RMS pilot_injection_hz close to
    peak / sqrt(2) -- the RMS of a pure sinusoid.

    Real FM stereo broadcasts target ~6.75 kHz peak pilot injection (9% of
    the 75 kHz full-scale reference); this test uses that exact value.
    """
    demod = _make_demodulator()
    duration = 0.05
    n = int(SR * duration)
    t = np.arange(n) / SR
    pilot_peak_hz = 6_750.0
    inst_freq = pilot_peak_hz * np.cos(2 * np.pi * 19_000.0 * t)

    iq = _fm_modulate(inst_freq, SR)
    audio, status = demod.demodulate(iq)

    assert status is not None
    expected_rms = pilot_peak_hz / np.sqrt(2)
    # Wider tolerance than the raw-deviation test: this value passes through
    # the pilot bandpass filter, whose passband ripple/rolloff has some
    # effect on measured amplitude, same as any real spectrum analyzer.
    assert status.pilot_injection_hz == pytest.approx(expected_rms, rel=0.15)
    # And it must actually be recognized as stereo, sanity-checking that
    # the synthesized signal is realistic enough to trip the existing
    # stereo_pilot_locked threshold too.
    assert status.stereo_pilot_locked


def test_rds_injection_filter_recovers_a_known_57khz_amplitude():
    """The RDS injection bandpass filter (built at __init__ when RBDS is
    enabled) must recover a known 57 kHz subcarrier amplitude, applied
    directly rather than through the full demodulate() -- avoids spinning
    up RBDSWorker's decode thread just to test filter gain/passband math.
    """
    demod = _make_demodulator(enable_rbds=True)
    assert demod._rds_injection_filter is not None

    duration = 0.05
    n = int(SR * duration)
    t = np.arange(n) / SR
    rds_peak_hz = 3_500.0  # within the typical ~2-4.5 kHz healthy range
    # multiplex-domain signal (already in Hz, matching what fm_discriminator
    # would output after the hz_per_radian_sample scale factor is undone --
    # apply the filter directly on this Hz-domain signal since that's what
    # the filter's frequency response is designed against).
    multiplex_hz = rds_peak_hz * np.cos(2 * np.pi * 57_000.0 * t)

    from scipy.signal import oaconvolve
    filtered = oaconvolve(multiplex_hz, demod._rds_injection_filter, mode="same")
    rms = float(np.sqrt(np.mean(filtered ** 2)))

    expected_rms = rds_peak_hz / np.sqrt(2)
    assert rms == pytest.approx(expected_rms, rel=0.15)


def test_rds_injection_filter_is_none_when_rbds_disabled():
    """No RBDS -> no RDS injection filter built at all -- confirms the
    receivers-that-don't-use-RDS-pay-nothing design intent."""
    demod = _make_demodulator(enable_rbds=False)
    assert demod._rds_injection_filter is None


def test_rds_injection_filter_rejects_the_pilot_band():
    """The RDS measurement filter must not leak significant energy from a
    strong 19 kHz pilot -- otherwise a normal stereo broadcast with no RDS
    at all would still read a nonzero, misleading RDS injection level."""
    demod = _make_demodulator(enable_rbds=True)
    n = int(SR * 0.05)
    t = np.arange(n) / SR
    pilot_only_hz = 6_750.0 * np.cos(2 * np.pi * 19_000.0 * t)

    from scipy.signal import oaconvolve
    filtered = oaconvolve(pilot_only_hz, demod._rds_injection_filter, mode="same")
    rms = float(np.sqrt(np.mean(filtered ** 2)))

    # Should be attenuated to a small fraction of the pilot's own RMS
    # (~4773 Hz) -- a strict near-zero isn't realistic for a FIR bandpass
    # ~38 kHz away from its passband, but it must be clearly suppressed.
    assert rms < 200.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
