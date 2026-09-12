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

"""The waterfall view, rendered from a captured IQ file."""

import json
import os
import time
import uuid
from typing import Any

from flask import Flask, jsonify, request

from app_core.models import RadioReceiver
from app_core.radio import (
    ensure_radio_tables,
)
from app_core.auth.roles import require_permission
from app_core.radio.decimation import effective_sample_rate

from . import deps


def register(app: Flask, route_logger) -> None:
    """Attach this module's routes to the app."""
    @app.route(
        "/api/radio/diagnostics/waterfall/<int:receiver_id>", methods=["POST"]
    )
    @require_permission('receivers.configure')
    def api_radio_diagnostics_waterfall(receiver_id: int) -> Any:
        """Capture a short IQ window and return a spectrogram + clipping stats.

        The endpoint reuses the existing ``capture_iq`` Redis command flow
        used by the "Capture IQ" button — same SDR-service path, same
        on-disk ``.npz`` format — but instead of handing the file back to
        the browser it loads the complex64 samples here, runs a
        ``scipy.signal.spectrogram`` on them, and returns a compact JSON
        payload the diagnostics page renders to a ``<canvas>``.

        The whole point is to make front-end overload visually obvious:
        a strong nearby station that is being clipped by the SDR's ADC
        shows up as wideband "fuzz" + a saturated peak/RMS, and the
        ``clipping_fraction`` field gives a numeric verdict.

        Request body (optional JSON):
            duration_sec: float, capture duration (default 0.5 s, max 5 s).
                The waterfall view uses a short window because we only
                need enough data to spot clipping/spurs, not to decode
                long-form metadata.

        On success returns:
            sample_rate, frequency_hz, num_samples, duration_sec,
            peak_dbfs, rms_dbfs, clipping_fraction,
            verdict (one of "ok", "warning", "clipping"),
            freqs_khz: list[float] length n_freq (offset from carrier),
            times_sec: list[float] length n_time,
            db_min, db_max: float, dBFS range covered by ``power_u8``,
            n_freq, n_time: int,
            power_u8: list[int] length n_time*n_freq, row-major
                (each row is one time slice). Quantised to uint8 so the
                JSON payload stays small (~64 KB for 256x256).
        """
        # Local imports keep the module's import-time cost low — these
        # are only needed when the operator clicks "Show Waterfall".
        try:
            import numpy as np
            from scipy import signal as scipy_signal
        except ImportError as exc:
            return jsonify({
                "error": "Waterfall requires numpy and scipy on the web host",
                "hint": str(exc),
            }), 503

        ensure_radio_tables(route_logger)
        receiver_record = RadioReceiver.query.get_or_404(receiver_id)

        payload = request.get_json(silent=True) or {}
        try:
            duration_sec = float(payload.get("duration_sec", 0.5))
        except (TypeError, ValueError):
            duration_sec = 0.5
        # Bound aggressively: the waterfall is a quick "is the front end
        # clipping right now" check, not a long-form recording. 5 s at
        # 2.4 MHz is already 96 MB of complex64 to FFT.
        duration_sec = max(0.05, min(duration_sec, 5.0))

        # Captures are tapped from the ring buffer, i.e. *after* early
        # decimation, so the requested duration must be converted at the
        # effective rate. Using the configured hardware rate here asked
        # for decim-factor times too many samples -- an Airspy at 2.5 MHz
        # (decim 10) turned a 5 s request into a 50 s capture that always
        # blew the wait budget below.
        effective_rate = effective_sample_rate(receiver_record.sample_rate) or 250_000
        num_samples = int(duration_sec * effective_rate)

        try:
            command_id = str(uuid.uuid4())
            redis_client = deps.get_redis_client()

            command = {
                "action": "capture_iq",
                "receiver_id": receiver_record.identifier,
                "command_id": command_id,
                "num_samples": num_samples,
            }

            route_logger.info(
                "Sending capture_iq for waterfall on receiver %s "
                "(num_samples=%d, command_id=%s)",
                receiver_record.identifier, num_samples, command_id,
            )
            redis_client.rpush("sdr:commands", json.dumps(command))

            timeout = duration_sec + deps.RADIO_CAPTURE_WAIT_SLACK_SEC
            start_time = time.time()
            result = None
            while time.time() - start_time < timeout:
                result_json = redis_client.get(
                    f"sdr:command_result:{command_id}"
                )
                if result_json:
                    result = json.loads(result_json)
                    break
                time.sleep(0.1)

            if not result:
                return jsonify({
                    "error": "Timeout waiting for sdr-service to write capture",
                    "hint": (
                        "Check if sdr-service is running: "
                        "sudo systemctl status eas-station-sdr.service"
                    ),
                }), 504

            if not result.get("success"):
                return jsonify({
                    "error": "Capture failed",
                    "hint": result.get("error", "Unknown error"),
                }), 503

            capture_path = result.get("path")
            if not capture_path or not deps._is_path_within(
                capture_path, deps.RADIO_CAPTURE_DIR
            ):
                # Same defence-in-depth as the download endpoint: refuse
                # to load an .npy from outside the allow-list directory.
                route_logger.error(
                    "Refusing waterfall on out-of-tree capture path: %s",
                    capture_path,
                )
                return jsonify({
                    "error": "Capture stored at unexpected location",
                }), 500
            if not os.path.isfile(capture_path):
                return jsonify({
                    "error": "Capture file disappeared before analysis",
                }), 500

            # Load + immediately delete the file.  The waterfall view is
            # ephemeral; we never re-serve the raw IQ here.  This matches
            # the single-use semantics of the download endpoint.
            #
            # The SDR service writes .npz with 'iq' (complex64) and
            # optionally 'multiplex' (see routes_diagnostics_analyze.py,
            # which handles this the same way) -- np.load() on a .npz
            # returns a lazy NpzFile container, not an array, so calling
            # .size on it directly raised AttributeError on every capture.
            # Fall back to a plain .npy of complex64 IQ for legacy captures.
            samples = None
            try:
                if capture_path.endswith(".npz"):
                    with np.load(capture_path, allow_pickle=False) as archive:
                        if "iq" in archive.files:
                            samples = archive["iq"]
                else:
                    samples = np.load(capture_path, allow_pickle=False)
            finally:
                try:
                    os.remove(capture_path)
                except OSError as cleanup_exc:
                    route_logger.debug(
                        "Failed to remove waterfall capture %s: %s",
                        capture_path, cleanup_exc,
                    )

            if samples is None or samples.size == 0:
                return jsonify({"error": "Capture contained no samples"}), 500

            # SoapySDR / libairspy drivers in this repo deliver normalised
            # complex64 with 0 dBFS == |x| == 1.0, so dBFS is meaningful
            # without any per-driver scaling.  See _measure_iq_levels in
            # sdr_hardware_service.py for the same convention.
            samples = np.asarray(samples, dtype=np.complex64)
            mag = np.abs(samples)
            peak = float(np.max(mag)) if mag.size else 0.0
            rms = (
                float(np.sqrt(np.mean(mag.astype(np.float64) ** 2)))
                if mag.size else 0.0
            )
            peak_dbfs = 20.0 * float(np.log10(peak)) if peak > 0 else -120.0
            rms_dbfs = 20.0 * float(np.log10(rms)) if rms > 0 else -120.0
            # The rest of the codebase treats |x| > 0.95 as "clipping" for
            # auto-gain decisions; reuse that threshold so the waterfall
            # verdict and the auto-gain knob agree on what "clipping"
            # means.  Anything above ~1% of samples sitting at the rails
            # is unambiguous front-end overload.
            clipping_fraction = float(np.mean(mag >= 0.95)) if mag.size else 0.0
            if clipping_fraction >= 0.01 or peak_dbfs >= -0.5:
                verdict = "clipping"
            elif clipping_fraction >= 0.001 or peak_dbfs >= -3.0:
                verdict = "warning"
            else:
                verdict = "ok"

            sample_rate = int(
                result.get("sample_rate") or effective_rate
            )

            # FFT sizing: 256 freq bins × ≤256 time slices keeps the JSON
            # response below ~80 KB once quantised to uint8 and is also
            # the right resolution for spotting wideband clipping vs. a
            # narrow spur on a typical browser canvas.
            n_fft = 256
            if samples.size < n_fft:
                # Zero-pad so we still get a single FFT row.  Bitter edge
                # case but easier than refusing the request.
                samples = np.concatenate([
                    samples,
                    np.zeros(n_fft - samples.size, dtype=np.complex64),
                ])
            noverlap = n_fft // 2
            # Stride down the time axis so we never produce more than 256
            # rows even for a 5-second / 2.4 MHz capture.
            target_time_slices = 256
            step = max(1, (samples.size - noverlap) // (n_fft - noverlap))
            stride = max(1, step // target_time_slices)

            freqs, times, sxx = scipy_signal.spectrogram(
                samples,
                fs=sample_rate,
                window="hann",
                nperseg=n_fft,
                noverlap=noverlap,
                return_onesided=False,  # complex IQ → two-sided spectrum
                scaling="spectrum",
                mode="magnitude",
            )
            # scipy returns freqs as 0..fs/2, -fs/2..0; rearrange so the
            # carrier (0 Hz baseband) sits in the middle of the array.
            sxx = np.fft.fftshift(sxx, axes=0)
            freqs = np.fft.fftshift(freqs)

            if stride > 1:
                sxx = sxx[:, ::stride]
                times = times[::stride]

            # Cap the time axis to ``target_time_slices`` even if the
            # stride math left a few extra columns.
            if sxx.shape[1] > target_time_slices:
                sxx = sxx[:, :target_time_slices]
                times = times[:target_time_slices]

            # Convert magnitude to dBFS.  With ``return_onesided=False``
            # and ``scaling="spectrum"`` the output magnitude per bin is
            # already on the same scale as the input, so 0 dBFS == 1.0
            # carries through directly.  Floor at -120 dBFS to avoid
            # log10(0) and to give the colour map a stable lower bound.
            with np.errstate(divide="ignore"):
                power_db = 20.0 * np.log10(np.maximum(sxx, 1e-12))
            db_max = float(np.max(power_db))
            # Use a 60 dB dynamic range below the peak so weak features
            # (the desired RBDS subcarrier vs. an overload spur) stay
            # visible without dragging the noise floor across the screen.
            db_min = db_max - 60.0
            power_db = np.clip(power_db, db_min, db_max)

            # Quantise to uint8 for transport.  Each cell becomes a
            # single byte; the client recovers dBFS as
            #   dbfs = db_min + (u8 / 255) * (db_max - db_min)
            scale = 255.0 / max(db_max - db_min, 1e-6)
            power_u8 = np.clip(
                np.round((power_db - db_min) * scale), 0, 255
            ).astype(np.uint8)
            # Row-major over (time, freq) so the client iterates the
            # outer loop over time slices and the inner over freq bins.
            power_u8_flat = power_u8.T.reshape(-1).tolist()

            n_freq = int(power_u8.shape[0])
            n_time = int(power_u8.shape[1])

            return jsonify({
                "sample_rate": sample_rate,
                "frequency_hz": result.get("frequency_hz"),
                "num_samples": int(samples.size),
                "duration_sec": float(samples.size) / float(sample_rate),
                "peak_dbfs": peak_dbfs,
                "rms_dbfs": rms_dbfs,
                "clipping_fraction": clipping_fraction,
                "verdict": verdict,
                "freqs_khz": [float(f) / 1000.0 for f in freqs.tolist()],
                "times_sec": [float(t) for t in times.tolist()],
                "db_min": float(db_min),
                "db_max": float(db_max),
                "n_freq": n_freq,
                "n_time": n_time,
                "power_u8": power_u8_flat,
            })

        except Exception as exc:
            route_logger.error(
                "Failed to compute waterfall for receiver %s: %s",
                receiver_id, exc, exc_info=True,
            )
            return jsonify({"error": "Failed to compute waterfall"}), 500


__all__ = ["register"]
