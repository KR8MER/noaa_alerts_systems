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

from __future__ import annotations

"""FM demodulator with stereo decoding and RBDS extraction."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import oaconvolve

from .dsp import (
    StreamingResampler,
    design_fir_bandpass,
    design_fir_lowpass,
    fast_decimate,
    fm_discriminator,
    fm_discriminator_declick,
    resample_to,
)
from .types import (
    DemodulatorConfig,
    DemodulatorStatus,
    RBDSData,
    RBDSDecoderStats,
)
from .rbds_worker import RBDSWorker

logger = logging.getLogger(__name__)

class FMDemodulator:
    """FM demodulator with stereo decoding and RBDS extraction.

    This demodulator uses a multi-stage decimation approach for high sample rate
    SDRs (like Airspy at 2.5 MHz). The signal processing chain is:

    1. IQ samples at SDR rate (e.g., 2.5 MHz)
    2. Phase discriminator to extract FM multiplex signal
    3. Decimate to intermediate rate (e.g., 250 kHz) for efficient filtering
    4. Apply audio lowpass filters at intermediate rate
    5. Stereo decode (if enabled) at intermediate rate
    6. Decimate/resample to final audio rate (e.g., 48 kHz)
    7. Apply de-emphasis filter

    This approach provides much better filter performance than trying to apply
    narrow audio filters directly at MHz sample rates.
    """

    # FM deviation constants for different modulation types
    # These determine the audio gain scaling factor
    FM_DEVIATION_HZ = {
        'WFM': 75000,   # Broadcast FM: ±75 kHz deviation
        'FM': 75000,    # Same as WFM
        'NFM': 5000,    # Narrowband FM: ±5 kHz deviation (NOAA, two-way radio)
    }

    # Default deviation for unknown modulation types (broadcast FM standard)
    DEFAULT_DEVIATION_HZ = 75000

    # Target intermediate sample rate for audio processing
    # 250 kHz is sufficient for FM stereo (needs > 76 kHz for 38 kHz subcarrier)
    # and provides good filter performance with reasonable tap counts
    INTERMEDIATE_SAMPLE_RATE = 250000

    def __init__(self, config: DemodulatorConfig):
        self.config = config

        # Normalize modulation type to uppercase for consistent lookup
        # This prevents issues with case sensitivity (fm vs FM vs Fm)
        self.config.modulation_type = config.modulation_type.upper()

        # Previous complex sample for phase continuity
        self._prev_sample: Optional[np.complex64] = None
        self._sample_index: int = 0

        # Magnitude-aware click suppressor state.  _declick_last_phase
        # carries the last good discriminator output across chunks so a
        # click at sample 0 of a new chunk doesn't have to wait for the
        # next good sample — same purpose as _prev_sample, but for the
        # post-discriminator domain.
        self._declick_last_phase: float = 0.0
        self._click_rate: float = 0.0

        # Calculate decimation factor for efficient processing
        # We want to get from SDR rate down to ~250 kHz for audio processing
        self._decimation_factor = 1
        self._intermediate_rate = config.sample_rate

        if config.sample_rate > self.INTERMEDIATE_SAMPLE_RATE * 2:
            # Calculate decimation to get close to target intermediate rate
            self._decimation_factor = max(1, int(config.sample_rate / self.INTERMEDIATE_SAMPLE_RATE))
            self._intermediate_rate = config.sample_rate // self._decimation_factor
            logger.info(
                "FM demodulator using %dx decimation: %d Hz -> %d Hz intermediate rate",
                self._decimation_factor, config.sample_rate, self._intermediate_rate
            )

        # Design decimation lowpass filter if needed
        # Cutoff at 80% of new Nyquist to prevent aliasing
        self._decim_filter = None
        if self._decimation_factor > 1:
            decim_cutoff = self._intermediate_rate * 0.4  # 40% of intermediate rate
            # More taps for better stopband rejection at high sample rates
            decim_taps = min(1024, max(256, config.sample_rate // 10000))
            self._decim_filter = self._design_fir_lowpass(decim_cutoff, config.sample_rate, taps=decim_taps)
            logger.debug("Decimation filter: %d taps, cutoff %.1f kHz", decim_taps, decim_cutoff / 1000)

        # Calculate FM audio gain based on modulation type and sample rate.
        # The discriminator output is the raw phase difference in radians:
        #   phase_diff_per_sample = 2π × deviation / sample_rate
        # so scaling by sample_rate / (2π × deviation) maps full deviation
        # to ±1.0.  (An earlier comment here claimed the discriminator
        # divides by π — it doesn't; fm_discriminator returns np.angle().)
        deviation_hz = self.FM_DEVIATION_HZ.get(self.config.modulation_type, self.DEFAULT_DEVIATION_HZ)
        self._audio_gain = config.sample_rate / (2.0 * np.pi * deviation_hz)

        # De-emphasis filter state
        self._deemph_alpha = 0.0
        if config.deemphasis_us > 0:
            tau = config.deemphasis_us * 1e-6
            self._deemph_alpha = 1.0 - np.exp(-1.0 / (config.audio_sample_rate * tau))
        self._deemph_state = np.zeros(1, dtype=np.float32)

        # Decimation phase carried across chunks by the mono audio path so
        # the every-Nth-sample stride stays continuous when a chunk length
        # isn't a multiple of the decimation factor.
        self._mono_decim_phase = 0
        # Overlap-add tail of the mono path's anti-alias FIR, carried
        # across chunks so the filter is seamless at chunk boundaries
        # (a per-chunk mode="same" convolution would put a ~2 ms zero-fed
        # transient at every boundary — an audible tick a few times/sec).
        self._mono_lpf_tail: Optional[np.ndarray] = None
        # Sample-continuous resamplers for the final intermediate→audio
        # rate conversion, keyed per channel ("mono"/"left"/"right").
        self._audio_resamplers: Dict[str, StreamingResampler] = {}

        # Stereo decoder state
        # Use intermediate rate for stereo processing (more efficient filters)
        self._stereo_enabled = (
            config.stereo_enabled
            and config.modulation_type in {"FM", "WFM"}
            and self._intermediate_rate >= 76000  # Minimum for 38kHz subcarrier
        )

        # Design audio filters for ORIGINAL sample rate since stereo/pilot detection
        # happens BEFORE decimation on the raw multiplex signal
        # CRITICAL FIX: Filters must match the sample rate of the signal they're applied to
        # The multiplex signal is at config.sample_rate, not intermediate_rate
        audio_filter_taps = self._calculate_filter_taps(16000.0, config.sample_rate)
        self._lpr_filter = self._design_fir_lowpass(16000.0, config.sample_rate, taps=audio_filter_taps)
        self._dsb_filter = self._design_fir_lowpass(16000.0, config.sample_rate, taps=audio_filter_taps)
        self._pilot_filter = self._design_fir_bandpass(18500.0, 19500.0, config.sample_rate, taps=audio_filter_taps)

        # Stereo carrier tracking state
        self._pilot_phase = 0.0
        self._pilot_freq = 19000.0  # 19 kHz pilot tone
        self._pilot_pll_bandwidth = 50.0  # Hz - narrow bandwidth for stable lock

        # RBDS processing in separate thread so it never blocks audio
        self._rbds_enabled = config.enable_rbds and config.sample_rate >= 114000
        self._rbds_worker: Optional[RBDSWorker] = None
        self._rbds_intermediate_rate = self._intermediate_rate

        # RDS subcarrier injection-level measurement -- distinct from (and
        # much cheaper than) RBDSWorker's own 54-60 kHz decode-chain filter:
        # this just measures how much of the composite deviation budget the
        # 57 kHz subcarrier occupies, mirroring exactly how self._pilot_filter
        # measures the 19 kHz pilot's injection level below. Only built when
        # RBDS is actually relevant, same gate as rbds_enabled, so receivers
        # that don't use RDS pay nothing extra.
        self._rds_injection_filter: Optional[np.ndarray] = (
            self._design_fir_bandpass(55600.0, 58400.0, config.sample_rate, taps=audio_filter_taps)
            if self._rbds_enabled else None
        )

        # Early-decimation state (PySDR architecture).  The RBDSWorker's
        # filter chain (54-60 kHz bandpass, 57 kHz mix, 2.4 kHz post-mix
        # lowpass, downstream decim to 25 kHz, then resample to 19 kHz) is
        # sized for ~250 kHz inputs — that's how PySDR's reference RDS
        # tutorial does it (https://pysdr.org/content/rds.html: starts at
        # sample_rate=250e3 and uses a 101-tap firwin lowpass).  Feeding the
        # worker raw SDR rate (e.g. 1 MHz, 2.4 MHz) leaves its 101-tap
        # bandpass with ~40-95 kHz transition vs a 6 kHz target passband —
        # essentially all-pass — and its 501-tap 2.4 kHz lowpass with ~10-25
        # kHz transition, so the 4 kHz post-mix stereo artifact lands in
        # the passband.  We must decimate the multiplex from the *actual,
        # user-configured* SDR rate down to _rbds_intermediate_rate before
        # submitting, with a proper anti-alias filter ahead of the
        # downsampler.  Initialised below only when RBDS is enabled.
        self._rbds_decim: int = 1
        self._rbds_aa_filter: Optional[np.ndarray] = None
        # Overlap-add convolution tail (see the call site below for why
        # this isn't an lfilter zi delay line).
        self._rbds_aa_tail: Optional[np.ndarray] = None
        # Sample-offset counter at the *decimated* rate.  The worker's
        # crystal-locked 57 kHz / pilot-locked 19 kHz reference uses
        # sample_offset / self._sample_rate to compute time, so the offset
        # must count samples at the rate the worker actually receives.
        self._rbds_sample_index: int = 0

        if self._rbds_enabled:
            # Pick the largest divisor of config.sample_rate that lands the
            # worker between 130 and ~280 kHz — the band where its filters
            # are correctly proportioned and post-decim Nyquist (>= 65 kHz)
            # safely covers the 57 kHz RBDS subcarrier.  This mirrors the
            # earlier branch that already existed for the audio path but
            # makes it the *actually-applied* rate for the RBDS path too.
            if config.sample_rate > self._intermediate_rate * 2:
                rbds_decim = config.sample_rate // self._intermediate_rate
                while (config.sample_rate // rbds_decim) < 130000 and rbds_decim > 1:
                    rbds_decim -= 1
                self._rbds_intermediate_rate = config.sample_rate // rbds_decim
                self._rbds_decim = rbds_decim
            else:
                # Low-rate SDR (e.g. user already running at 250 kHz):
                # multiplex is already at a reasonable rate, no early
                # decimation needed.  rbds_decim stays at 1.
                self._rbds_intermediate_rate = config.sample_rate
                self._rbds_decim = 1

            # Build the rate-adaptive anti-alias filter when we're going to
            # decimate.  Cutoff must (a) preserve the 57 kHz RBDS subcarrier
            # (whose upper edge is 60 kHz) and (b) put the stopband edge
            # below post-decim Nyquist so nothing folds back into the RBDS
            # band.  Tap count scales with the input rate to hold the
            # transition bandwidth roughly constant — the cap of 1025
            # caps CPU even at Airspy's 10 MHz native rate while keeping
            # the stopband under post-decim Nyquist for any rate the user
            # is realistically going to set.
            if self._rbds_decim > 1:
                post_decim_nyquist = self._rbds_intermediate_rate / 2.0
                # See drivers.py:_initialize_sample_buffer for the same
                # rationale: the FM signal's spectral shoulders extend
                # to ±~95-100 kHz, so an 80 kHz cutoff (the old value)
                # clips full-deviation peaks and breaks the constant-
                # envelope assumption, which smears the 38 kHz and
                # 57 kHz subcarriers.  Open the cutoff up to whatever
                # post-decim Nyquist allows (minus a 15 kHz transition
                # band), capped at 110 kHz and floored at 80 kHz so
                # the prior contract is honoured on narrow post-decim
                # rates that can't safely go wider.
                transition_band = max(15_000.0, post_decim_nyquist * 0.15)
                rbds_aa_cutoff = min(
                    post_decim_nyquist - transition_band,
                    110_000.0,
                )
                rbds_aa_cutoff = max(80_000.0, rbds_aa_cutoff)
                # Tap-count heuristic: scale linearly with input rate to
                # hold transition bandwidth roughly constant.  Dividing by
                # 4000 gives ~250 taps at 1 MHz and ~600 taps at 2.4 MHz,
                # which yields a transition bandwidth of ~16 kHz — wide
                # enough that the stopband edge sits comfortably below
                # post-decim Nyquist for every realistic SDR rate.  The
                # `| 1` forces an odd number, required for a Type-I linear-
                # phase symmetric FIR (so group delay is an integer number
                # of samples).  Floor of 127 keeps low-rate users from
                # getting a useless filter; cap of 1025 bounds CPU at
                # Airspy's 10 MHz native rate.
                rbds_aa_taps = max(127, min(1025, (int(config.sample_rate / 4000) | 1)))
                self._rbds_aa_filter = self._design_fir_lowpass(
                    rbds_aa_cutoff, config.sample_rate, taps=rbds_aa_taps
                )
                logger.info(
                    "RBDS anti-alias filter: %d taps, cutoff %.1f kHz @ %d Hz "
                    "→ %d Hz (decim=%d), post-decim Nyquist=%.1f kHz",
                    rbds_aa_taps,
                    rbds_aa_cutoff / 1000.0,
                    config.sample_rate,
                    self._rbds_intermediate_rate,
                    self._rbds_decim,
                    post_decim_nyquist / 1000.0,
                )

            # Create RBDS worker thread - all processing happens there.
            # CRITICAL: pass the *effective* sample rate the worker will
            # receive (post early-decimation), NOT the raw SDR rate.  Its
            # internal filters (bandpass / pilot bandpass / 2.4 kHz post-
            # mix lowpass) are designed against this value.
            self._rbds_worker = RBDSWorker(
                self._rbds_intermediate_rate,
                self._rbds_intermediate_rate,
            )
            logger.info(
                "RBDS ENABLED: worker at %d Hz (input sample_rate=%d Hz, decim=%d)",
                self._rbds_intermediate_rate,
                config.sample_rate,
                self._rbds_decim,
            )
        else:
            # Log clearly why RBDS is not enabled
            if not config.enable_rbds:
                logger.info(
                    "RBDS DISABLED: enable_rbds=False in receiver config"
                )
            elif config.sample_rate < 114000:
                logger.info(
                    "RBDS DISABLED: sample_rate=%d Hz is below 114 kHz minimum",
                    config.sample_rate
                )
            else:
                logger.info(
                    "RBDS DISABLED: enable_rbds=%s, sample_rate=%d Hz",
                    config.enable_rbds,
                    config.sample_rate
                )


    def _calculate_filter_taps(self, cutoff_hz: float, sample_rate: int, transition_bw_ratio: float = 0.125) -> int:
        """Calculate appropriate number of filter taps for given parameters.

        Args:
            cutoff_hz: Filter cutoff frequency in Hz
            sample_rate: Sample rate in Hz
            transition_bw_ratio: Transition bandwidth as fraction of cutoff (default 12.5%)

        Returns:
            Number of filter taps (odd number for symmetric filter)
        """
        # Transition bandwidth
        transition_bw = cutoff_hz * transition_bw_ratio

        # Kaiser formula approximation: taps ≈ (stopband_attenuation_dB - 8) / (2.285 * transition_bw_normalized)
        # For ~60 dB stopband attenuation:
        # taps ≈ (60 - 8) / (2.285 * (transition_bw / sample_rate)) = 52 / (2.285 * transition_bw / sample_rate)
        taps = int(52.0 * sample_rate / (2.285 * transition_bw))

        # Ensure odd number and reasonable range
        taps = max(65, min(1025, taps | 1))  # Clamp to 65-1025, ensure odd

        return taps

    def process(self, iq_samples: np.ndarray) -> np.ndarray:
        """
        Process IQ samples and return audio samples.

        This is the main entry point used by audio processing pipeline.
        Demodulator status (RBDS, stereo pilot) is available via get_last_status().

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Audio samples (float32 numpy array)
        """
        audio, status = self.demodulate(iq_samples)
        self._last_status = status  # Store for get_last_status()
        return audio

    def get_last_status(self) -> Optional[DemodulatorStatus]:
        """Get the most recent demodulator status (stereo pilot, RBDS data)."""
        return getattr(self, '_last_status', None)

    def reset_rbds(self) -> None:
        """Drop all RBDS state (sync, filters, decoded metadata).

        Call this when tuning to a new station.  Without it, the decoder
        keeps showing the previous station's PS/radiotext until the new
        station's first group gets decoded, and the carrier-phase / symbol
        timing state from the old station delays re-locking.
        """
        if self._rbds_worker is not None:
            self._rbds_worker.reset()
        # Also drop the last-status reference to the old station's RBDS
        # data so any downstream consumer that reads get_last_status()
        # before the next demodulate() sees a clean slate.
        last = getattr(self, '_last_status', None)
        if last is not None:
            last.rbds_data = None
            last.rbds_synced = False

    def is_rbds_synced(self) -> bool:
        """True once the RBDS bit-level sync state machine has locked."""
        if self._rbds_worker is None:
            return False
        return self._rbds_worker.is_synced()

    def demodulate(self, iq_samples: np.ndarray) -> Tuple[np.ndarray, Optional[DemodulatorStatus]]:
        """
        Demodulate FM signal from IQ samples.

        Optimized for real-time processing at high IQ sample rates (2.5 MHz).
        Optionally extracts RBDS data and detects stereo pilot tone.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Tuple of (audio samples, demodulator status with RBDS/stereo info)
        """
        if len(iq_samples) == 0:
            return np.array([], dtype=np.float32), None

        # FAST PATH: Using JIT-compiled functions when available
        iq_array = np.asarray(iq_samples, dtype=np.complex64)

        # Mean IQ magnitude - raw RF signal strength for the RSSI meter.  Done
        # on the input array before phase-continuity prepending so the value
        # reflects the samples that just arrived from the SDR.
        rf_signal_strength = float(np.mean(np.abs(iq_array))) if iq_array.size else 0.0

        # Phase continuity - prepend last sample from previous block
        if self._prev_sample is not None:
            iq_array = np.concatenate(([self._prev_sample], iq_array))
        self._prev_sample = iq_array[-1]

        # Phase discriminator - uses Numba JIT if available (50-100x faster)
        # This is the core FM demodulation algorithm
        # Output is the FM multiplex signal containing L+R, stereo (L-R at 38kHz), and RBDS (at 57kHz)
        #
        # On a Class B station at 7 mi with light multipath the *signal*
        # is plenty strong, but the IQ envelope dips momentarily during
        # fades and atan2 emits uniform phase noise during those dips.
        # That noise spreads as a flat impulse floor across the whole
        # MPX and buries the 57 kHz RBDS subcarrier.  When the user
        # enables click suppression we use the magnitude-aware variant
        # which forward-fills the last good phase during envelope
        # collapses.
        if self.config.enable_click_suppression and self.config.click_suppression_threshold > 0.0:
            multiplex, click_count, self._declick_last_phase = fm_discriminator_declick(
                iq_array,
                suppression_fraction=self.config.click_suppression_threshold,
                prev_phase=self._declick_last_phase,
            )
            self._click_rate = (
                click_count / len(multiplex) if len(multiplex) > 0 else 0.0
            )
        else:
            multiplex = fm_discriminator(iq_array)
            self._click_rate = 0.0

        # Detect stereo pilot tone (19 kHz) and RBDS extraction
        # Must happen BEFORE audio decimation destroys the subcarriers
        rbds_data: Optional[RBDSData] = None
        stereo_pilot_locked = False
        stereo_pilot_strength = 0.0
        pilot_filtered: Optional[np.ndarray] = None
        pilot_rms: Optional[float] = None

        # Hz-per-radian-per-sample: the inverse of _audio_gain (which maps
        # raw discriminator output to normalized audio where the reference
        # deviation_hz = amplitude 1.0). multiplex and any band-limited
        # slice of it are still in raw radians/sample; multiplying by this
        # factor converts straight to real Hz, no separate calibration
        # needed -- FM demodulation of a digital SDR is trustworthy on
        # frequency even without amplitude calibration.
        hz_per_radian_sample = self.config.sample_rate / (2.0 * np.pi)

        # Peak instantaneous deviation of the whole composite MPX signal
        # this chunk (FCC full-scale reference: +/-75 kHz). Free -- multiplex
        # is already computed above regardless of stereo/RBDS being enabled.
        peak_deviation_hz = (
            float(np.max(np.abs(multiplex))) * hz_per_radian_sample
            if multiplex.size else 0.0
        )
        pilot_injection_hz = 0.0
        rds_injection_hz = 0.0

        # Stereo pilot detection (19 kHz tone indicates stereo broadcast)
        if self._stereo_enabled and self.config.sample_rate >= 38000:
            # Filter for 19 kHz pilot tone
            pilot_filtered = oaconvolve(multiplex, self._pilot_filter, mode="same")

            # Measure pilot strength (RMS of filtered signal)
            pilot_rms = float(np.sqrt(np.mean(pilot_filtered ** 2)))
            stereo_pilot_strength = min(1.0, pilot_rms * 10.0)  # Scale to 0-1 range
            # Calibrated Hz reading (typical healthy target: ~6.75 kHz, 9%
            # of 75 kHz full-scale) -- reuses pilot_rms, no extra filtering.
            pilot_injection_hz = pilot_rms * hz_per_radian_sample

            # Pilot is considered "locked" if strength exceeds threshold
            stereo_pilot_locked = stereo_pilot_strength > 0.1  # 10% threshold

            if stereo_pilot_locked:
                logger.debug("Stereo pilot detected: strength=%.2f", stereo_pilot_strength)

        # RDS subcarrier injection level (typical healthy range ~2-4.5 kHz).
        # Same technique as the pilot measurement above, gated the same way
        # RBDS decoding itself is gated -- see self._rds_injection_filter's
        # construction in __init__ for why this costs nothing on receivers
        # that don't use RDS.
        if self._rds_injection_filter is not None:
            rds_filtered = oaconvolve(multiplex, self._rds_injection_filter, mode="same")
            rds_rms = float(np.sqrt(np.mean(rds_filtered ** 2)))
            rds_injection_hz = rds_rms * hz_per_radian_sample

        # RBDS extraction in a separate worker thread.  Submit samples
        # (non-blocking) and pick up whatever the worker has decoded since
        # the last call.
        # 24/7 RELIABILITY: Wrap in try-except to ensure RBDS issues never affect audio
        if self._rbds_enabled and self._rbds_worker:
            try:
                # Early-decimate the multiplex from the user-configured SDR
                # rate down to the worker's intermediate rate (PySDR
                # architecture).  When _rbds_decim == 1 (low-rate SDR or
                # already-decimated input) we skip the filter and pass the
                # multiplex through untouched — same as the legacy code
                # path.
                if self._rbds_decim > 1 and self._rbds_aa_filter is not None:
                    from scipy import signal as scipy_signal
                    # Overlap-add via oaconvolve, carrying the convolution
                    # tail across chunks so the filter's history survives
                    # the boundary -- NOT lfilter: a pure-FIR lfilter call
                    # (a=1.0, this filter's case) unconditionally takes
                    # scipy's O(N*taps) direct-form np.convolve fallback
                    # regardless of dtype or zi; the fast C path only
                    # activates for a true IIR filter (len(a) > 1). Same
                    # bug, same fix as app_core/radio/drivers.py and
                    # RBDSWorker._process_rbds -- found via a codebase-wide
                    # audit for this exact pattern after it turned up twice
                    # already.
                    full = scipy_signal.oaconvolve(multiplex, self._rbds_aa_filter)
                    tail = self._rbds_aa_tail
                    if tail is not None and tail.size:
                        if tail.size > full.size:
                            full = np.concatenate(
                                [full, np.zeros(tail.size - full.size, dtype=full.dtype)]
                            )
                        full[: tail.size] += tail
                    n_in = len(multiplex)
                    self._rbds_aa_tail = full[n_in:].copy()
                    filtered = full[:n_in]
                    # Decimate by integer factor.  Anti-aliasing was just
                    # done above so the [::N] is safe — same pattern PySDR
                    # uses (firwin → np.convolve → x[::10]).
                    rbds_multiplex = filtered[:: self._rbds_decim].astype(np.float32)
                else:
                    rbds_multiplex = multiplex

                # Pass the absolute sample offset *in the worker's domain*
                # so its crystal-locked 57 kHz / pilot-locked 19 kHz
                # reference stays phase-coherent across chunks even when
                # chunks are dropped from the queue (queue overflow
                # discards chunks in the audio thread).  This counter is
                # decoupled from self._sample_index, which counts
                # multiplex samples at the raw SDR rate for any callers
                # that depend on that semantic (e.g. tests).
                self._rbds_worker.submit_samples(rbds_multiplex, self._rbds_sample_index)
                self._rbds_sample_index += len(rbds_multiplex)

                # Get whatever RBDS data is available (may be from previous chunks)
                rbds_data = self._rbds_worker.get_latest_data()
            except Exception as e:
                # Log but don't let RBDS errors affect audio demodulation
                logger.warning("RBDS error (audio unaffected): %s", e)

        # Advance the absolute sample index AFTER submitting to ensure the
        # offset is the position of the FIRST sample in this chunk.
        self._sample_index += len(multiplex)

        # Calculate decimation factor for audio downsampling
        target_rate = self.config.audio_sample_rate
        decim = max(1, self.config.sample_rate // target_rate)

        # Stereo decoding - must happen BEFORE decimation destroys the 38 kHz subcarrier
        # The L-R difference signal is modulated at 38 kHz (double the 19 kHz pilot)
        stereo_audio = None
        if self._stereo_enabled and stereo_pilot_locked and self.config.sample_rate >= 76000:
            # _decode_stereo's sample_indices param is unused by the method
            # body (kept only for backwards-compat) -- building it here used
            # to allocate a full-chunk-length float64 array every stereo-
            # locked chunk for nothing.  Pass pilot_filtered/pilot_rms
            # instead, which the method *does* use.
            try:
                stereo_audio = self._decode_stereo(
                    multiplex, pilot_filtered=pilot_filtered, pilot_rms=pilot_rms
                )
                if stereo_audio is not None:
                    logger.debug("Stereo decoded: %d samples, shape %s", len(stereo_audio), stereo_audio.shape)
            except Exception as e:
                logger.warning("Stereo decoding error: %s", e, exc_info=True)
                stereo_audio = None
        # CRITICAL FIX: Use proper resampling to exact target rate instead of simple decimation
        # Simple decimation produces wrong sample rate: e.g., 2.5MHz / 52 = 48,077 Hz (not 48,000 Hz)
        # This causes "chipmunk" audio when played back at declared rate
        if stereo_audio is not None:
            # We have stereo audio - decimate and resample both channels
            if decim > 1:
                # Decimate each channel separately
                left = fast_decimate(stereo_audio[:, 0], decim)
                right = fast_decimate(stereo_audio[:, 1], decim)
                intermediate_rate = self.config.sample_rate // decim
                logger.debug(
                    f"FM stereo demod: IQ {self.config.sample_rate}Hz → decim {decim}x → "
                    f"{intermediate_rate}Hz → resample → {target_rate}Hz"
                )
            else:
                left = stereo_audio[:, 0]
                right = stereo_audio[:, 1]
                intermediate_rate = self.config.sample_rate

            # Scale to audio levels: full deviation → ±1.0.  The box
            # decimation above has unity DC gain, so the factor must not
            # depend on decim (the old /decim under-drove the audio).
            left = left * self._audio_gain
            right = right * self._audio_gain

            # Resample to exact target rate
            if intermediate_rate != target_rate:
                left = self._stream_resample(left, intermediate_rate, target_rate, "left")
                right = self._stream_resample(right, intermediate_rate, target_rate, "right")

            audio = np.column_stack((left, right))
        else:
            # Mono audio path.  Bandlimit the raw multiplex with the same
            # proper 16 kHz FIR the stereo decoder uses for L+R before any
            # decimation.  The old box-average decimation (fast_decimate)
            # had only ~10-20 dB of stopband, so the 19 kHz pilot, the
            # 38 kHz L-R subcarrier, and the 57 kHz RBDS carrier all
            # folded straight into the audible band.
            audio = self._mono_audio_lowpass(multiplex)
            if decim > 1:
                # Every-Nth-sample stride is alias-safe now that the FIR
                # above removed everything past the post-decimation
                # Nyquist.  Carry the stride phase across chunks so no
                # samples are dropped at chunk boundaries (fast_decimate
                # silently discarded len % decim samples per chunk,
                # causing a small periodic time skip).
                audio = audio[self._mono_decim_phase::decim]
                self._mono_decim_phase = (self._mono_decim_phase - len(multiplex)) % decim
                intermediate_rate = self.config.sample_rate // decim
                logger.debug(
                    f"FM demod: IQ {self.config.sample_rate}Hz → LPF+decim {decim}x → "
                    f"{intermediate_rate}Hz → resample → {target_rate}Hz"
                )
            else:
                intermediate_rate = self.config.sample_rate
                logger.debug("FM demod: No decimation needed, %sHz → %sHz", intermediate_rate, target_rate)

            # Scale to audio levels BEFORE resampling: discriminator output
            # is 2π × f_dev / sample_rate rad/sample, so _audio_gain
            # (sample_rate / (2π × deviation)) maps full deviation to ±1.0.
            audio = audio * self._audio_gain

            # Now resample from intermediate_rate to exact target_rate
            # This ensures audio is at the EXACT sample rate expected by downstream consumers
            if intermediate_rate != target_rate:
                audio = self._stream_resample(audio, intermediate_rate, target_rate, "mono")
                logger.debug(
                    f"Resampled {len(audio)} samples from {intermediate_rate}Hz to {target_rate}Hz"
                )

        # De-emphasis — undo the transmitter's pre-emphasis (75 µs NA,
        # 50 µs EU).  Must run at the final audio rate because
        # _deemph_alpha was derived from audio_sample_rate in __init__.
        # Without this the treble sits up to ~17 dB hot and sounds like
        # clipping distortion even though no limiter is engaging.
        audio = self._apply_deemphasis(audio)

        # Clamp to prevent overflow
        audio = np.clip(audio, -1.5, 1.5)

        # Soft-clip to prevent harsh distortion on overmodulated signals
        # Uses tanh with reduced gain for smoother limiting
        # Scale down before tanh and back up after to preserve dynamics
        audio = np.tanh(audio * 0.7) / 0.7

        # Create demodulator status with stereo pilot and RBDS info
        decoder_stats: Optional[RBDSDecoderStats] = None
        if self._rbds_enabled and self._rbds_worker is not None:
            decoder_stats = self._rbds_worker.get_stats()
        status = DemodulatorStatus(
            rbds_data=rbds_data,
            stereo_pilot_locked=stereo_pilot_locked,
            stereo_pilot_strength=stereo_pilot_strength,
            is_stereo=self._stereo_enabled and stereo_pilot_locked,
            signal_strength=rf_signal_strength,
            rbds_synced=(
                self._rbds_worker.is_synced()
                if self._rbds_enabled and self._rbds_worker is not None
                else False
            ),
            rbds_enabled=self._rbds_enabled,
            rbds_decoder_stats=decoder_stats,
            click_rate=self._click_rate,
            click_suppression_enabled=(
                bool(self.config.enable_click_suppression)
                and self.config.click_suppression_threshold > 0.0
            ),
            peak_deviation_hz=peak_deviation_hz,
            pilot_injection_hz=pilot_injection_hz,
            rds_injection_hz=rds_injection_hz,
        )

        return audio.astype(np.float32), status

    @staticmethod
    def _resample(signal: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Thin shim around the module-level :func:`resample_to`."""
        return resample_to(signal, from_rate, to_rate)

    def _stream_resample(
        self, audio: np.ndarray, from_rate: int, to_rate: int, key: str
    ) -> np.ndarray:
        """Resample via a persistent per-channel :class:`StreamingResampler`."""
        resampler = self._audio_resamplers.get(key)
        if (
            resampler is None
            or resampler.from_rate != from_rate
            or resampler.to_rate != to_rate
        ):
            resampler = StreamingResampler(from_rate, to_rate)
            self._audio_resamplers[key] = resampler
        return resampler.process(audio)

    def _mono_audio_lowpass(self, multiplex: np.ndarray) -> np.ndarray:
        """Anti-alias FIR for the mono audio path, seamless across chunks.

        Streaming overlap-add: convolve the chunk in "full" mode (FFT
        based, so the 1k-tap filter stays cheap even at MHz rates), add
        the tail carried over from the previous chunk, emit the first
        len(chunk) samples, and keep the new tail for the next call.
        This is bit-identical to filtering one continuous stream, unlike
        a per-chunk mode="same" convolution which feeds zeros at every
        chunk edge.  The causal filter delays audio by taps//2 samples
        (~2 ms at 256 kHz) which is irrelevant for broadcast monitoring.
        """
        full = oaconvolve(multiplex, self._lpr_filter)
        tail = self._mono_lpf_tail
        if tail is not None and tail.size:
            if tail.size > full.size:
                # Chunk shorter than the filter tail — extend so no
                # carried samples are lost.
                full = np.concatenate(
                    [full, np.zeros(tail.size - full.size, dtype=full.dtype)]
                )
            full[: tail.size] += tail
        n = len(multiplex)
        self._mono_lpf_tail = full[n:].copy()
        return full[:n]

    def _apply_deemphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply de-emphasis filter (single-pole IIR lowpass).

        Vectorized via scipy.signal.lfilter; the filter delay line is kept
        in ``self._deemph_state`` so the response is continuous across
        chunk boundaries.  y[n] = y[n-1] + alpha * (x[n] - y[n-1]).
        """
        if self._deemph_alpha <= 0.0 or audio.size == 0:
            return audio

        from scipy import signal as scipy_signal

        b = [self._deemph_alpha]
        a = [1.0, self._deemph_alpha - 1.0]

        channels = 1 if audio.ndim == 1 else audio.shape[1]
        if self._deemph_state.shape[0] != channels:
            # Channel count changed mid-stream (stereo pilot lock acquired
            # or lost) — restart the delay line from silence.
            self._deemph_state = np.zeros(channels, dtype=np.float32)

        if audio.ndim == 1:
            output, zf = scipy_signal.lfilter(
                b, a, audio, zi=self._deemph_state.astype(np.float64)
            )
            self._deemph_state = zf.astype(np.float32)
        else:
            output, zf = scipy_signal.lfilter(
                b, a, audio, axis=0,
                zi=self._deemph_state[np.newaxis, :].astype(np.float64),
            )
            self._deemph_state = zf[0].astype(np.float32)
        return output.astype(audio.dtype, copy=False)

    @staticmethod
    def _design_fir_lowpass(cutoff: float, fs: int, taps: int = 129) -> np.ndarray:
        """Thin shim around the module-level :func:`design_fir_lowpass`."""
        return design_fir_lowpass(cutoff, fs, taps=taps)

    @staticmethod
    def _design_fir_bandpass(low_cut: float, high_cut: float, fs: int, taps: int = 129) -> np.ndarray:
        """Thin shim around the module-level :func:`design_fir_bandpass`."""
        return design_fir_bandpass(low_cut, high_cut, fs, taps=taps)

    def _lpr_filter_signal(self, signal: np.ndarray) -> np.ndarray:
        filtered = oaconvolve(signal, self._lpr_filter, mode="same")
        return filtered

    def _decode_stereo(
        self,
        multiplex: np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
        pilot_filtered: Optional[np.ndarray] = None,
        pilot_rms: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Decode FM stereo from multiplex signal.

        Derives a phase-coherent 38 kHz carrier directly from the recovered
        19 kHz pilot tone by squaring it (cos²(ωt) = ½(1 + cos(2ωt))).  This
        approach is automatically locked to the transmitter's pilot phase and
        frequency, so it tracks both crystal drift and chunk-boundary phase
        without any PLL state.  An earlier implementation generated a free-
        running 38 kHz oscillator that reset to phase 0 on every chunk; with
        the SDR clock differing from the broadcaster's by tens of ppm the L-R
        signal slowly rotated against the carrier, collapsing the stereo image
        toward mono and bleeding L into R.

        Args:
            multiplex: FM multiplex signal (after discriminator)
            sample_indices: Unused; kept as an optional param purely for
                backwards-compat with any external caller passing it
                positionally. ``demodulate()`` no longer builds this array
                -- it used to allocate a full-length ``np.arange(..., dtype=
                float64)`` every stereo-locked chunk (same size as the raw
                IQ chunk -- tens of MB/sec worth of churn at typical SDR
                rates) purely to satisfy this parameter, which the method
                body never reads. Confirmed via grep across this file that
                no code path consumes it.
            pilot_filtered: The 19 kHz-bandpass-filtered multiplex, if the
                caller already computed it (``demodulate()`` always has --
                it needs the same value to decide ``stereo_pilot_locked``
                *before* even calling this method). Confirmed via py-spy on
                a live, CPU-contended box that recomputing this identical
                ``oaconvolve`` call here -- same ``multiplex``, same
                ``self._pilot_filter`` -- was pure duplicate work, a third
                of this method's three FFT convolutions for zero benefit.
                Computed internally when omitted (e.g. every direct-call
                test in tests/test_fm_stereo_decoder.py), so this stays a
                pure optimization with no behavior change either way.
            pilot_rms: The RMS of ``pilot_filtered``, if the caller already
                computed it. ``demodulate()`` always has -- it needs the
                same value to compute ``stereo_pilot_strength`` before even
                calling this method, so recomputing the identical mean+sqrt
                reduction here was the same class of duplicate work as
                ``pilot_filtered`` above (smaller cost -- O(N) vs the FFT
                convolution -- but still literally the same reduction over
                the same array). Computed internally when omitted.

        Returns:
            Stereo audio as Nx2 array (left, right) or None if stereo cannot
            be decoded for this chunk.
        """
        if not self._stereo_enabled or len(multiplex) == 0:
            return None

        # Extract L+R (mono) using lowpass filter
        lpr = oaconvolve(multiplex, self._lpr_filter, mode="same")

        # Recover the 19 kHz pilot tone via the pre-designed bandpass --
        # unless the caller already has it (see the docstring above).
        if pilot_filtered is None:
            pilot_filtered = oaconvolve(multiplex, self._pilot_filter, mode="same")

        # Normalize the pilot to ~unit amplitude so the derived 38 kHz carrier
        # has a stable amplitude and the L-R recovery gain doesn't depend on
        # signal strength.  RMS · √2 is the peak amplitude of a sinusoid.
        if pilot_rms is None:
            pilot_rms = float(np.sqrt(np.mean(pilot_filtered ** 2)))
        pilot_peak = pilot_rms * np.sqrt(2.0)
        if pilot_peak < 1e-9:
            # Pilot vanished mid-chunk; let the caller fall back to mono.
            return None
        pilot_unit = pilot_filtered / pilot_peak

        # cos²(ωt) = ½(1 + cos(2ωt)) → 2·cos²(ωt) − 1 = cos(2ωt)
        # This is an exact, phase-coherent 38 kHz reference that tracks the
        # actual pilot frequency (no PLL or oscillator state needed).
        carrier_38k = 2.0 * (pilot_unit * pilot_unit) - 1.0

        # Demodulate L-R: mix with the 38 kHz carrier and lowpass.  The 2× gain
        # compensates for the ½ factor that DSB-SC mixing of unit-amplitude
        # signals introduces (cos(A)·cos(A) = ½(1 + cos(2A))).
        suppressed = 2.0 * multiplex * carrier_38k
        lmr = oaconvolve(suppressed, self._dsb_filter, mode="same")

        # Matrix decode: L = ½((L+R) + (L-R)), R = ½((L+R) - (L-R))
        left = 0.5 * (lpr + lmr)
        right = 0.5 * (lpr - lmr)

        return np.column_stack((left, right))

    def stop(self) -> None:
        """Stop the demodulator and clean up resources."""
        if self._rbds_worker:
            self._rbds_worker.stop()
            self._rbds_worker = None

