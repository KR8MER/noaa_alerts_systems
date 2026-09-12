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

Regression tests for the live spectrum frequency axis.

Reported live on the SDR Diagnostics page: an FM broadcast station tuned
at 93.9 MHz filled a Live Waterfall / Spectrum Scope axis labelled
93.388-94.412 MHz -- a 1.024 MHz span. A US FM channel is 200 kHz wide,
so the picture was physically impossible.

Root cause: the FFT behind both views runs on samples pulled from the
ring buffer, which sit *after* the early-decimation stage (see
app_core/radio/decimation.py). On a receiver configured for 1.024 MHz
the decimator runs at factor 4, so those samples are clocked at 256 kHz
and the bins cover 256 kHz of RF. But the SDR service never published
freq_min/freq_max, so the web route fell back to
``receiver.frequency_hz +/- receiver.sample_rate / 2`` -- the *configured
hardware* rate -- and drew the axis 4x too wide. A ~200 kHz signal
occupying ~78% of a 256 kHz span was therefore labelled as occupying
~20% of a megahertz.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_core.radio.decimation import (
    EARLY_DECIM_TARGET_RATE,
    early_decimation_factor,
    effective_sample_rate,
)
from webapp.radio_settings.routes_signal import _spectrum_axis


class _FakeReceiver:
    """Minimal stand-in for a RadioReceiver row."""

    def __init__(self, sample_rate, frequency_hz=93_900_000, driver="airspy"):
        self.sample_rate = sample_rate
        self.frequency_hz = frequency_hz
        self.driver = driver


# --------------------------------------------------------------------------
# Decimation math
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "configured, expected_factor, expected_effective",
    [
        (250_000, 1, 250_000),          # at target: untouched
        (500_000, 1, 500_000),          # exactly 2x target: still untouched
        (1_024_000, 4, 256_000),        # the reported receiver
        (2_400_000, 9, 266_666),        # RTL-SDR max
        (2_500_000, 10, 250_000),       # Airspy
        (10_000_000, 40, 250_000),      # Airspy high rate
    ],
)
def test_effective_rate_matches_driver_math(
    configured, expected_factor, expected_effective
):
    assert early_decimation_factor(configured) == expected_factor
    assert effective_sample_rate(configured) == expected_effective


def test_decimation_helpers_tolerate_missing_rate():
    for bad in (None, 0, "", "not-a-number"):
        assert early_decimation_factor(bad) == 1
        assert effective_sample_rate(bad) == 0


def test_driver_uses_the_shared_helper():
    """The driver must not carry its own copy of the decimation math."""
    from app_core.radio import drivers

    source = pathlib.Path(drivers.__file__).with_suffix(".py").read_text()
    assert "early_decimation_factor(self.config.sample_rate)" in source
    assert "effective_sample_rate(self.config.sample_rate)" in source
    assert drivers.EARLY_DECIM_TARGET_RATE == EARLY_DECIM_TARGET_RATE


# --------------------------------------------------------------------------
# The axis itself -- the actual reported bug
# --------------------------------------------------------------------------

def test_axis_uses_published_span_not_configured_rate():
    """The reported case: 1.024 MHz configured, 256 kHz actually covered."""
    receiver = _FakeReceiver(sample_rate=1_024_000)
    payload = {
        "sample_rate": 256_000,          # what the service publishes
        "hardware_sample_rate": 1_024_000,
        "early_decim_factor": 4,
        "center_frequency": 93_900_000,
        "freq_min": 93_900_000 - 128_000,
        "freq_max": 93_900_000 + 128_000,
    }

    axis = _spectrum_axis(payload, receiver)

    span = axis["freq_max"] - axis["freq_min"]
    assert span == 256_000
    # The bug drew this span; assert we never regress to it.
    assert span != 1_024_000
    assert axis["freq_min"] == 93_772_000
    assert axis["freq_max"] == 94_028_000
    assert axis["hardware_sample_rate"] == 1_024_000
    assert axis["early_decim_factor"] == 4


def test_axis_recomputes_span_when_service_omits_it():
    """An older sdr-service publishes no freq_min/freq_max/decim factor.

    The route must still land on the effective span rather than falling
    back to the configured hardware rate, which is what produced the
    original 4x-too-wide axis.
    """
    receiver = _FakeReceiver(sample_rate=1_024_000)

    axis = _spectrum_axis({"center_frequency": 93_900_000}, receiver)

    assert axis["sample_rate"] == 256_000
    assert axis["early_decim_factor"] == 4
    assert axis["freq_max"] - axis["freq_min"] == 256_000


def test_fm_broadcast_occupies_a_plausible_fraction_of_the_span():
    """A 200 kHz FM channel must not fit inside a fraction of the axis.

    This is the operator-facing sanity check from the bug report: on the
    broken axis a full-bandwidth FM signal appeared to be ~1 MHz wide.
    """
    receiver = _FakeReceiver(sample_rate=1_024_000)
    axis = _spectrum_axis({"center_frequency": 93_900_000}, receiver)

    fm_channel_hz = 200_000
    span = axis["freq_max"] - axis["freq_min"]
    occupancy = fm_channel_hz / span

    # ~78% of the 256 kHz window. On the broken 1.024 MHz axis this was
    # ~20%, meaning the visible signal had to be read as 1 MHz wide.
    assert 0.7 <= occupancy <= 0.85


def test_axis_handles_untouched_low_rate_receiver():
    """Below the decimation threshold, configured rate == span."""
    receiver = _FakeReceiver(sample_rate=250_000)

    axis = _spectrum_axis({"center_frequency": 162_400_000}, receiver)

    assert axis["sample_rate"] == 250_000
    assert axis["early_decim_factor"] == 1
    assert axis["freq_max"] - axis["freq_min"] == 250_000


def test_axis_survives_missing_centre_frequency():
    receiver = _FakeReceiver(sample_rate=1_024_000, frequency_hz=None)

    axis = _spectrum_axis({}, receiver)

    assert axis["freq_min"] is None
    assert axis["freq_max"] is None
    assert axis["sample_rate"] == 256_000


def test_axis_survives_empty_payload_and_unconfigured_receiver():
    receiver = _FakeReceiver(sample_rate=None, frequency_hz=None)

    axis = _spectrum_axis(None, receiver)

    assert axis["sample_rate"] == 0
    assert axis["freq_min"] is None
    assert axis["freq_max"] is None


# --------------------------------------------------------------------------
# What the SDR service publishes
# --------------------------------------------------------------------------

def test_service_publishes_the_effective_span():
    import sdr_hardware_service as svc

    payload = svc.build_spectrum_payload(
        identifier="wxj93",
        spectrum=[0.5] * 8,
        effective_rate=256_000,
        center_frequency=93_900_000,
        hardware_sample_rate=1_024_000,
        early_decim_factor=4,
        timestamp=1234.0,
    )

    assert payload["freq_min"] == 93_772_000
    assert payload["freq_max"] == 94_028_000
    assert payload["freq_max"] - payload["freq_min"] == 256_000
    assert payload["sample_rate"] == 256_000
    assert payload["hardware_sample_rate"] == 1_024_000
    assert payload["early_decim_factor"] == 4
    assert payload["status"] == "available"


def test_service_payload_survives_missing_centre_frequency():
    import sdr_hardware_service as svc

    payload = svc.build_spectrum_payload(
        identifier="wxj93",
        spectrum=[],
        effective_rate=256_000,
        center_frequency=None,
    )

    assert payload["freq_min"] is None
    assert payload["freq_max"] is None
    assert payload["early_decim_factor"] == 1


def test_published_payload_round_trips_through_the_route_helper():
    """End to end: what the service publishes is what the axis renders."""
    import sdr_hardware_service as svc

    payload = svc.build_spectrum_payload(
        identifier="wxj93",
        spectrum=[0.5] * 8,
        effective_rate=256_000,
        center_frequency=93_900_000,
        hardware_sample_rate=1_024_000,
        early_decim_factor=4,
    )
    axis = _spectrum_axis(payload, _FakeReceiver(sample_rate=1_024_000))

    assert axis["freq_min"] == payload["freq_min"]
    assert axis["freq_max"] == payload["freq_max"]
    assert axis["freq_max"] - axis["freq_min"] == 256_000


# --------------------------------------------------------------------------
# Capture duration -> sample count
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name",
    [
        "webapp.radio_settings.routes_diagnostics_waterfall",
        "webapp.radio_settings.routes_diagnostics_capture",
        "webapp.radio_settings.routes_diagnostics_analyze",
    ],
)
def test_capture_endpoints_convert_duration_at_effective_rate(module_name):
    """Captures tap the ring buffer, so duration converts at the effective rate.

    Converting at the configured rate asked for decim-factor times too
    many samples -- an Airspy at 2.5 MHz turned a 5 s request into a 50 s
    capture, which always exceeded the wait budget.
    """
    import importlib

    module = importlib.import_module(module_name)
    source = pathlib.Path(module.__file__).read_text()
    assert "effective_sample_rate(receiver_record.sample_rate)" in source
    assert "effective_rate = receiver_record.sample_rate or 250_000" not in source


# --------------------------------------------------------------------------
# Live waterfall / scope zoom invariants
# --------------------------------------------------------------------------
#
# The zoom feature rests on a few non-obvious properties of the template's
# JavaScript that are easy to undo by accident. These assert the contract
# in the spirit of tests/test_map_theme.py, which likewise guards template
# behaviour that has no Python surface.

DIAGNOSTICS_HTML = ROOT / "templates" / "admin" / "radio_diagnostics.html"


def _diagnostics_source() -> str:
    # Explicit encoding: the platform default is ASCII under a C/POSIX
    # locale, which would raise UnicodeDecodeError on the template's
    # non-ASCII characters and make the result runner-dependent.
    return DIAGNOSTICS_HTML.read_text(encoding="utf-8")


def test_waterfall_history_stores_raw_values_not_pixels():
    """The scrollback buffer must hold per-bin power, not finished RGBA.

    Zoom re-renders the whole history at the current crop. If the buffer
    goes back to storing rasterised pixels, zoomed history degrades to
    upscaled blocks -- it cannot recover detail that was already flattened
    to screen resolution when the row was stored.
    """
    src = _diagnostics_source()
    assert re.search(r"state\.buffer\s*=\s*new\s+Uint8Array\(", src), (
        "waterfall history must be a Uint8Array of per-bin power"
    )
    assert not re.search(
        r"state\.buffer\s*=\s*new\s+Uint8ClampedArray\(", src
    ), "waterfall history must not go back to storing RGBA pixels"


def test_zoom_is_bounded_by_real_fft_resolution():
    """The UI must not offer zoom past the resolution the FFT actually has."""
    src = _diagnostics_source()
    assert "const ZOOM_MAX =" in src
    assert "const ZOOM_MIN_BINS =" in src
    # zoomRange must clamp the window inside [0, nFreq].
    assert "Math.min(nFreq - count, start)" in src


def test_axis_and_status_follow_the_visible_window():
    """Labels must describe the crop, not the full span.

    A zoom that leaves the axis reading the full span would recreate the
    exact class of bug this page already shipped once: a frequency label
    that does not match the picture under it.
    """
    src = _diagnostics_source()
    assert "function visibleFreqRange(state, payload)" in src
    # Both views render their axis through the shared, zoom-aware helper.
    # Assert each call site by its distinct third argument: a bare count of
    # "renderSpectrumAxis(state, payload," also matches the declaration, so
    # it stayed >= 2 even if one of the two call sites were deleted.
    assert "renderSpectrumAxis(state, payload, state.axisEl)" in src
    assert "renderSpectrumAxis(state, payload, state.scopeAxisEl)" in src
    # The span readout takes the zoom state so it can report the crop.
    assert "function spectrumSpanLabel(payload, state, nFreq)" in src


def test_scope_peak_hold_is_indexed_by_absolute_bin():
    """Panning must not discard peaks accumulated off-screen."""
    src = _diagnostics_source()
    assert "state.scopePeak = new Float32Array(spectrum);" in src
    assert "for (let i = binStart; i < binEnd; i++)" in src


def test_zoom_gestures_leave_vertical_page_scroll_alone():
    """On a phone, claiming both axes would trap the page scroll."""
    src = _diagnostics_source()
    assert "canvas.style.touchAction = 'pan-y';" in src


# --------------------------------------------------------------------------
# Fixes from code review on PR #2416
# --------------------------------------------------------------------------

def test_null_frequency_bounds_are_rejected_not_coerced_to_zero():
    """``Number(null)`` is 0 and passes ``Number.isFinite``.

    ``build_spectrum_payload`` emits ``freq_min``/``freq_max`` as ``None``
    when the centre frequency cannot be coerced, which serialises to JSON
    ``null``. A guard that only checks ``Number.isFinite`` therefore lets
    the null through and labels both axis edges ``0.00000 MHz`` across a
    zero-width span. The explicit null check must stay.
    """
    src = _diagnostics_source()
    assert "if (payload.freq_min == null || payload.freq_max == null) return null;" in src
    # A degenerate (zero or inverted) span is rejected too.
    assert "fMax <= fMin" in src


def test_wheel_only_cancels_the_page_scroll_when_it_zooms():
    """Cancelling unconditionally trapped the page at both zoom limits.

    At 1x every scroll-down and at ZOOM_MAX every scroll-up produced no
    zoom, so an unconditional ``preventDefault`` left the operator unable
    to scroll the page while the pointer sat over a full-width canvas.
    """
    src = _diagnostics_source()
    wheel = src.split("addEventListener('wheel'", 1)[1].split("}, { passive: false });", 1)[0]
    # preventDefault must be inside the "did it zoom?" branch.
    assert "if (zoomBy(state, factor, anchor)) {" in wheel
    guarded = wheel.split("if (zoomBy(state, factor, anchor)) {", 1)[1]
    assert "ev.preventDefault();" in guarded
    # ...and must not appear before the zoom is evaluated.
    before = wheel.split("if (zoomBy(state, factor, anchor)) {", 1)[0]
    assert "ev.preventDefault();" not in before


def test_scope_autoscale_includes_the_peak_hold_trace():
    """scaleY() clamps to [lo, hi]; excluding the peaks pinned them flat.

    The peak-hold line retains maxima above the current spectrum, so a
    y-range computed from the live trace alone clipped the amber trace
    against the top edge instead of scaling to show it.
    """
    src = _diagnostics_source()
    assert "state.scopePeak ? Number(state.scopePeak[i]) : NaN" in src
    assert "if (isFinite(pk) && pk > hi) hi = pk;" in src


def test_status_lines_are_built_in_one_place():
    """The poll tick and a zoom refresh must not build the text separately."""
    src = _diagnostics_source()
    assert "function waterfallStatusText(state, payload, n, ageMs)" in src
    assert "function scopeStatusText(state, payload, n, ageMs)" in src
    # Two call sites each: the poll tick and zoomRefreshStatus.
    assert src.count("waterfallStatusText(state,") == 3   # decl + 2 calls
    assert src.count("scopeStatusText(state,") == 3


def test_pan_repaints_are_coalesced_to_one_per_frame():
    """A fast drag must not queue more repaints than the display can show."""
    src = _diagnostics_source()
    assert "requestAnimationFrame(() => {" in src
    assert "panFrame = 0;" in src


def test_zoom_controls_seed_from_live_state():
    """Panels rebuild ~1/s; a hardcoded 1x contradicted the zoomed canvas."""
    src = _diagnostics_source()
    assert "label.textContent = zoomLabelText(state);" in src
    assert "reset.disabled = !zoomIsActive(state);" in src


# --------------------------------------------------------------------------
# Snapshot Waterfall: .npz vs .npy capture format
# --------------------------------------------------------------------------
#
# Caught live by actually clicking "Snapshot Waterfall": every capture
# failed with a 500 -- "AttributeError: 'NpzFile' object has no attribute
# 'size'". sdr_hardware_service.py writes captures as .npz (np.savez, key
# 'iq'; see routes_diagnostics_analyze.py's comment on the same format),
# but routes_diagnostics_waterfall.py called np.load(path).size directly,
# which only works for a plain .npy array -- np.load() on a .npz returns a
# lazy NpzFile container with no .size attribute at all.

def test_waterfall_route_handles_npz_capture_format():
    """The real, actually-written capture format (.npz) must not crash
    the route the way a bare np.load(path).size did in production."""
    import numpy as np
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        npz_path = f"{tmp}/capture.npz"
        np.savez(npz_path, iq=np.ones(1024, dtype=np.complex64))

        # The exact extraction routes_diagnostics_waterfall.py now uses.
        samples = None
        with np.load(npz_path, allow_pickle=False) as archive:
            if "iq" in archive.files:
                samples = archive["iq"]

        assert samples is not None
        assert samples.size == 1024  # would have raised AttributeError before the fix

        # Document the actual bug: np.load() on a .npz without indexing
        # into it returns an NpzFile, not an array.
        raw = np.load(npz_path, allow_pickle=False)
        try:
            with pytest.raises(AttributeError):
                raw.size  # noqa: B018 - intentional attribute-access to reproduce the bug
        finally:
            raw.close()


def test_waterfall_route_source_reads_npz_before_checking_size():
    """Guard against the fix being reverted to a bare np.load(path).size."""
    from webapp.radio_settings import routes_diagnostics_waterfall

    src = pathlib.Path(routes_diagnostics_waterfall.__file__).read_text()
    assert 'capture_path.endswith(".npz")' in src
    assert 'archive["iq"]' in src
