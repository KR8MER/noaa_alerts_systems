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

"""
Redis SDR Source Adapter

Subscribes to Redis pub/sub to receive already-demodulated audio from
eas-station-demod.service, and provides that audio to the audio controller.

This is the bridge between the demod service (FM/AM demodulation) and
audio-service (audio processing + EAS monitoring) in the separated
architecture:

    sdr-service (IQ capture)
        -> demod-service (FM/AM demod: app_core/radio/demod, RBDS, ...)
            -> audio-service (this file: Icecast injection, EAS monitor)

Demodulation used to run inline in this adapter's own Redis-subscriber
thread. A ``py-spy record --gil`` profile of the live audio service showed
that thread dominating GIL-held time -- several ``scipy.signal.oaconvolve``
FFT convolutions per IQ chunk (stereo pilot detection, RBDS extraction --
see ``app_core/radio/demod/fm.py``) were starving the three real-time
Icecast feeder threads sharing the same process/GIL, which needed to wake
roughly every 50ms to keep their buffers fed. That DSP work now runs in
``eas-station-demod.service``, its own OS process, where it can never
again share a GIL with anything real-time. See
``docs/architecture/SDR_SERVICE_ARCHITECTURE.md`` and
``services/demod/worker.py`` for the full design.
"""

import base64
import json
import logging
import queue
import threading
import time
from typing import Optional, Any

import numpy as np
import redis.exceptions

from .ingest import AudioSourceAdapter, AudioSourceConfig, AudioSourceStatus, AudioMetrics

logger = logging.getLogger(__name__)

#: How long a fetched demod status is reused before re-fetching from Redis.
#: _update_metrics() runs roughly once per audio chunk (~tens of ms), and
#: RBDS/stereo status changes far less often than that -- refetching on
#: every call would just be needless Redis round-trips for data that is,
#: almost always, identical to what was fetched a moment ago.
_STATUS_CACHE_TTL_S = 0.25


def _unpack_audio_envelope(payload: bytes) -> "tuple[int, int, np.ndarray]":
    """Inverse of ``services.demod.worker._pack_audio_envelope``.

    Duplicated here (rather than imported from ``services.demod``) on
    purpose: ``app_core`` is the shared library every service and the
    webapp import *from*; nothing under ``app_core`` should import from
    a ``services.*`` package, so a three-line pure function is kept local
    instead of reaching across that boundary.

    Returns ``(iq_sample_rate, center_frequency, audio_samples)``.
    """
    iq_sample_rate = int.from_bytes(payload[:4], "big", signed=False)
    center_frequency = int.from_bytes(payload[4:8], "big", signed=False)
    audio = np.frombuffer(payload[8:], dtype=np.float32)
    return iq_sample_rate, center_frequency, audio


class RedisSDRSourceAdapter(AudioSourceAdapter):
    """
    Audio source adapter that receives demodulated audio from Redis pub/sub.

    Subscribes to ``demod:audio:{receiver_id}`` (published by
    eas-station-demod.service) and feeds that audio to the broadcast
    queue. Decoder status (stereo lock, RBDS PS/PI/radiotext, ...) is
    read from the ``demod:status:{receiver_id}`` key the demod service
    refreshes alongside it.
    """

    def __init__(self, config: AudioSourceConfig):
        super().__init__(config)
        self._redis_client: Optional[Any] = None
        self._pubsub: Optional[Any] = None
        # Note: self._audio_queue is created by base class via BroadcastQueue subscription
        # Don't override it - use self._source_broadcast.publish() instead
        self._receiver_id: Optional[str] = None
        self._last_sample_time: float = 0.0
        self._samples_received: int = 0
        self._iq_sample_rate: int = 2500000  # Informational only now; demod owns the real value
        self._center_frequency: int = 0  # Populated from the demod status snapshot, see below
        # Queue for audio chunks from Redis subscriber thread
        self._audio_chunk_queue: queue.Queue = queue.Queue(maxsize=100)
        # Last-known RBDS data so cleared-on-this-cycle fields keep displaying
        # the most recent decoded value until something new arrives, and so the
        # `rbds_last_updated` timestamp only advances when data actually changes.
        self._rbds_data: Optional[Any] = None
        self._rbds_signature: Optional[tuple] = None
        # Cache for _get_remote_status() -- see _STATUS_CACHE_TTL_S.
        self._status_cache: Optional[Any] = None
        self._status_cache_at: float = 0.0

    def _start_capture(self) -> None:
        """Start Redis subscription to the demod service's audio channel."""
        # Get receiver ID from config
        self._receiver_id = self.config.device_params.get('receiver_id')
        if not self._receiver_id:
            raise ValueError("receiver_id required in device_params for Redis SDR source")

        # Connect to Redis
        from app_core.redis_client import get_redis_client
        try:
            self._redis_client = get_redis_client()
            logger.info(f"Connected to Redis for receiver {self._receiver_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}") from e

        from app_core.config.redis_config import RedisChannels

        # Subscribe to the demod service's output channel for this receiver.
        self._pubsub = self._redis_client.pubsub(ignore_subscribe_messages=True)
        channel = f"{RedisChannels.DEMOD_AUDIO_PREFIX}{self._receiver_id}"
        self._pubsub.subscribe(channel)
        logger.info(f"Subscribed to Redis channel: {channel}")

        # Start Redis subscriber thread (separate from capture thread).
        # This thread is now deliberately thin -- it only unpacks the
        # audio envelope and enqueues it, no DSP -- so it can never become
        # the same kind of GIL hog the old inline-demod version was.
        subscriber_thread = threading.Thread(
            target=self._redis_subscriber_loop,
            name=f"redis-sdr-{self._receiver_id}",
            daemon=True
        )
        subscriber_thread.start()
        logger.info(f"Started Redis SDR subscriber for {self._receiver_id}")

    def _redis_subscriber_loop(self) -> None:
        """Redis pub/sub subscriber loop - receives demodulated audio."""
        logger.info(f"Redis subscriber loop started for {self._receiver_id}")

        last_log_time = time.time()
        messages_received = 0

        try:
            # Use _stop_event from base class instead of undefined _running
            while not self._stop_event.is_set():
                try:
                    # Use get_message with timeout instead of listen() to allow graceful shutdown
                    # Check pubsub availability inside try block to avoid race conditions
                    pubsub = self._pubsub
                    if pubsub is None:
                        logger.debug(f"Redis pubsub closed for {self._receiver_id}, exiting subscriber loop")
                        break

                    message = pubsub.get_message(timeout=1.0)
                except (OSError, ConnectionError, redis.exceptions.ConnectionError) as e:
                    # Handle connection errors gracefully (e.g., socket closed during shutdown)
                    # OSError with errno 9 = Bad file descriptor (socket was closed)
                    # ConnectionError = Built-in connection reset by peer
                    # redis.exceptions.ConnectionError = Redis-specific connection errors (e.g., server closed connection)
                    if self._stop_event.is_set():
                        # Expected during shutdown - log at debug level
                        logger.debug(f"Redis connection closed during shutdown for {self._receiver_id}: {e}")
                    else:
                        # Unexpected connection error - log as error
                        logger.error(f"Redis connection error for {self._receiver_id}: {e}")
                    break

                # Log periodically if no samples received
                current_time = time.time()
                if message is None:
                    if current_time - last_log_time > 10.0:
                        if messages_received == 0:
                            logger.warning(
                                f"No demodulated audio received for {self._receiver_id} in "
                                f"{current_time - last_log_time:.1f}s. Check that "
                                f"eas-station-demod.service is running and receiving IQ samples "
                                f"for this receiver."
                            )
                        last_log_time = current_time
                    continue

                if message.get('type') != 'message':
                    continue

                try:
                    messages_received += 1
                    raw = message['data']
                    if not raw:
                        logger.warning(f"Empty audio envelope from demod for {self._receiver_id}")
                        continue
                    # base64-decoded, not raw bytes -- see the matching
                    # comment in services/demod/worker.py::DemodWorker._publish
                    # for why: the shared Redis client decodes every
                    # pub/sub payload as UTF-8, which crashes outright on
                    # a raw binary payload before this code ever runs.
                    if isinstance(raw, bytes):
                        raw = raw.decode('ascii')
                    payload = base64.b64decode(raw)
                    if len(payload) < 8:
                        logger.warning(f"Short audio envelope from demod for {self._receiver_id}")
                        continue

                    sample_rate, center_frequency, audio_samples = _unpack_audio_envelope(payload)
                    self._iq_sample_rate = sample_rate
                    if center_frequency:
                        self._center_frequency = center_frequency

                    # _unpack_audio_envelope always returns a flat 1-D array --
                    # the demod worker publishes np.column_stack((left, right))
                    # for stereo, which .tobytes() flattens to interleaved
                    # L,R,L,R floats, and np.frombuffer() on the way back out
                    # can't recover that (frames, channels) shape on its own.
                    # Without reshaping here, IcecastOutputStreamer's
                    # _samples_to_pcm_bytes() sees a 1-D array and treats every
                    # one of the 2x-as-many flat values as an independent mono
                    # sample, then upmixes each to fake stereo -- doubling the
                    # frame count it hands FFmpeg. FFmpeg still thinks it's
                    # getting config.sample_rate frames/sec, so the encoded
                    # stream ends up representing ~2x as much declared audio
                    # duration per real second of broadcast, which plays back
                    # at roughly half speed (deep, dragging pitch) and destroys
                    # the actual L/R stereo image in the process.
                    channels = max(1, int(self.config.channels))
                    if (
                        channels > 1
                        and audio_samples.ndim == 1
                        and audio_samples.size % channels == 0
                    ):
                        audio_samples = audio_samples.reshape(-1, channels)

                    if audio_samples is not None and len(audio_samples) > 0:
                        # Put audio in queue for _read_audio_chunk() to consume
                        # The base class capture loop will handle metrics updates and broadcasting
                        try:
                            self._audio_chunk_queue.put(audio_samples, timeout=0.1)
                            self._samples_received += len(audio_samples)
                            self._last_sample_time = time.time()

                            # Log first successful sample
                            if messages_received == 1:
                                logger.info(
                                    f"✅ First audio chunk received for {self._receiver_id}: "
                                    f"{len(audio_samples)} samples"
                                )
                        except queue.Full:
                            logger.warning(f"Audio chunk queue full for {self._receiver_id}, dropping samples")

                except Exception as e:
                    logger.error(f"Error processing demodulated audio message: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Redis subscriber loop error: {e}", exc_info=True)
        finally:
            logger.info(
                f"Redis subscriber loop exited for {self._receiver_id}. "
                f"Processed {messages_received} messages"
            )

    def _get_remote_status(self):
        """Fetch+decode the demod service's latest DemodulatorStatus.

        Replaces the old ``self._demodulator.get_last_status()`` call now
        that the demodulator lives in a different process. Cached briefly
        (see ``_STATUS_CACHE_TTL_S``) since ``_update_metrics()`` runs far
        more often than the status meaningfully changes. Returns ``None``
        on any failure (Redis down, key expired/absent, decode error) --
        callers already treat "no status" as "nothing decoded yet", the
        same as the old in-process path when a demodulator hadn't produced
        a status yet.
        """
        now = time.monotonic()
        if self._status_cache is not None and (now - self._status_cache_at) < _STATUS_CACHE_TTL_S:
            return self._status_cache

        if self._redis_client is None or not self._receiver_id:
            return None

        try:
            from app_core.config.redis_config import RedisChannels
            from app_core.radio.demod.types import demodulator_status_from_json_dict

            raw = self._redis_client.get(f"{RedisChannels.DEMOD_STATUS_PREFIX}{self._receiver_id}")
            if raw is None:
                return self._status_cache  # keep last-known rather than flapping to None on a TTL gap
            # The writer (services/demod/worker.py::DemodWorker._publish)
            # writes plain JSON text here, not pickle -- pickle.loads on a
            # value read from a shared store is an arbitrary-code-execution
            # primitive the moment anything untrusted can write to this key.
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            status = demodulator_status_from_json_dict(json.loads(raw))
            self._status_cache = status
            self._status_cache_at = now
            return status
        except Exception as exc:
            logger.debug(f"Failed to fetch demod status for {self._receiver_id}: {exc}")
            return self._status_cache

    def _stop_capture(self) -> None:
        """Stop Redis subscription."""
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception as e:
                # Log at debug level since errors during cleanup are expected
                logger.debug(f"Error closing Redis pub/sub for {self._receiver_id}: {e}")
            finally:
                # Mark pubsub as None to signal subscriber loop to exit
                self._pubsub = None

        # Clear audio chunk queue
        while not self._audio_chunk_queue.empty():
            try:
                self._audio_chunk_queue.get_nowait()
            except queue.Empty:
                break

        logger.info(f"Stopped Redis SDR source for {self._receiver_id}")

    def _read_audio_chunk(self) -> Optional[np.ndarray]:
        """Read an audio chunk from the queue filled by Redis subscriber thread.
        
        This method is called by the base class's capture loop.
        The Redis subscriber thread demodulates IQ samples and puts audio in the queue.
        """
        try:
            # Get audio chunk from queue with short timeout
            # This allows the capture loop to check _stop_event periodically
            audio_chunk = self._audio_chunk_queue.get(timeout=0.1)
            return audio_chunk
        except queue.Empty:
            # No audio available yet - this is normal, return None
            return None

    def _update_metrics(self, audio_chunk: Optional[np.ndarray] = None) -> None:
        """Update metrics from Redis SDR source."""
        super()._update_metrics(audio_chunk)

        # Add Redis-specific metadata
        if self.metrics.metadata is None:
            self.metrics.metadata = {}

        self.metrics.metadata['source_type'] = 'redis_sdr'
        self.metrics.metadata['receiver_id'] = self._receiver_id
        self.metrics.metadata['iq_sample_rate'] = self._iq_sample_rate
        self.metrics.metadata['center_frequency'] = self._center_frequency
        self.metrics.metadata['receiver_frequency_hz'] = self._center_frequency  # For waterfall display
        demod_mode = self.config.device_params.get('demod_mode', 'FM')
        self.metrics.metadata['receiver_modulation'] = demod_mode  # For waterfall display
        self.metrics.metadata['demodulation_enabled'] = True  # For waterfall display
        self.metrics.metadata['center_frequency_mhz'] = round(self._center_frequency / 1_000_000, 6)
        self.metrics.metadata['samples_received'] = self._samples_received
        self.metrics.metadata['last_sample_age'] = time.time() - self._last_sample_time if self._last_sample_time > 0 else None

        # Stereo detection from demodulator
        # stereo_enabled = modulation type supports stereo (WFM/FM)
        # stereo_pilot_locked = 19kHz pilot tone actually detected
        # is_stereo = receiving stereo audio (pilot locked and stereo mode)
        modulation_supports_stereo = demod_mode.upper() in ('WFM', 'FM')
        self.metrics.metadata['stereo_enabled'] = modulation_supports_stereo

        # Get actual stereo detection status from the demod service (see
        # _get_remote_status() -- the demodulator itself now lives in
        # eas-station-demod.service, not this process).
        status = self._get_remote_status()
        if status:
            self.metrics.metadata['stereo_pilot_locked'] = status.stereo_pilot_locked
            self.metrics.metadata['stereo_pilot_strength'] = status.stereo_pilot_strength
            self.metrics.metadata['is_stereo'] = status.is_stereo
            # Override stereo_enabled with actual detection for display
            if modulation_supports_stereo:
                self.metrics.metadata['stereo_enabled'] = status.stereo_pilot_locked

            # RBDS lock state.  Lets the UI render a LOCKING / LOCKED /
            # DISABLED badge so users know whether a missing PS/RT means
            # no data yet (still locking), the decoder isn't even trying
            # (disabled in receiver config), or the modulation isn't FM.
            rbds_synced = bool(getattr(status, 'rbds_synced', False))
            rbds_enabled_runtime = bool(getattr(status, 'rbds_enabled', False))
            self.metrics.metadata['rbds_synced'] = rbds_synced
            self.metrics.metadata['rbds_enabled'] = rbds_enabled_runtime
            if rbds_synced:
                lock_state = 'LOCKED'
            elif not modulation_supports_stereo:
                lock_state = 'UNAVAILABLE'
            elif not rbds_enabled_runtime:
                # FM modulation but RBDS decoding is off — without this
                # branch the UI showed "Acquiring sync…" indefinitely
                # for receivers configured with enable_rbds=False.
                lock_state = 'DISABLED'
            else:
                lock_state = 'LOCKING'
            self.metrics.metadata['rbds_lock_state'] = lock_state

            decoder_stats = getattr(status, 'rbds_decoder_stats', None)
            if decoder_stats is not None:
                self.metrics.metadata['rbds_blocks_total'] = decoder_stats.blocks_total
                self.metrics.metadata['rbds_blocks_ok'] = decoder_stats.blocks_ok
                self.metrics.metadata['rbds_blocks_fec_single'] = decoder_stats.blocks_fec_single
                self.metrics.metadata['rbds_blocks_fec_burst'] = decoder_stats.blocks_fec_burst
                self.metrics.metadata['rbds_blocks_uncorrected'] = decoder_stats.blocks_uncorrected
                self.metrics.metadata['rbds_blocks_bit_slips'] = decoder_stats.blocks_bit_slips
                self.metrics.metadata['rbds_groups_decoded'] = decoder_stats.groups_decoded
                self.metrics.metadata['rbds_sync_acquired_unix'] = decoder_stats.sync_acquired_unix
                self.metrics.metadata['rbds_sync_lost_count'] = decoder_stats.sync_lost_count
                self.metrics.metadata['rbds_chunks_dropped'] = decoder_stats.chunks_dropped
                self.metrics.metadata['rbds_raw_bler'] = decoder_stats.raw_block_error_rate
                self.metrics.metadata['rbds_net_bler'] = decoder_stats.net_block_error_rate
                self.metrics.metadata['rbds_group_type_counts'] = (
                    dict(decoder_stats.group_type_counts)
                    if decoder_stats.group_type_counts else None
                )
                # Field-churn / false-read telemetry from the two-sighting
                # confirmation gate (see RBDSDecoderStats).
                self.metrics.metadata['rbds_pi_change_count'] = decoder_stats.pi_change_count
                self.metrics.metadata['rbds_pty_change_count'] = decoder_stats.pty_change_count
                self.metrics.metadata['rbds_ta_toggle_count'] = decoder_stats.ta_toggle_count
                self.metrics.metadata['rbds_glitches_rejected'] = decoder_stats.glitches_rejected

            # RF RSSI (mean IQ magnitude).  Linear value; the UI converts to
            # dBFS for the signal meter.  Without this the RSSI indicator is
            # permanently blank on Redis-backed SDR sources.
            if getattr(status, 'signal_strength', None) is not None:
                self.metrics.metadata['rf_signal_strength'] = float(status.signal_strength)
                self.metrics.metadata['rf_signal_strength_updated'] = time.time()

            # Multipath/impulse-noise indicator -- fraction of discriminator
            # samples the click suppressor replaced this chunk (see
            # DemodulatorStatus.click_rate). Not persisted anywhere before
            # this: _snapshot_audio_metrics_once() dumps the whole metadata
            # dict once/sec, so adding it here is enough to backfill the
            # signal-quality history endpoint with no other plumbing.
            if getattr(status, 'click_rate', None) is not None:
                self.metrics.metadata['click_rate'] = float(status.click_rate)
                self.metrics.metadata['click_suppression_enabled'] = bool(
                    getattr(status, 'click_suppression_enabled', False)
                )

            # Extract RBDS/RDS data if available.  We cache the last decoded
            # object so that between decoder poll cycles we can keep showing
            # the most recent values (RBDS groups arrive every ~100 ms, so a
            # given _update_metrics() call very often sees the same object
            # as the previous one — we must not restamp rbds_last_updated in
            # that case, otherwise "last updated" looks fresh forever).
            rbds = status.rbds_data
            from .sources import RBDS_PROGRAM_TYPES
            if rbds is not None:
                signature = (
                    rbds.ps_name,
                    rbds.pi_code,
                    rbds.call_sign,
                    rbds.pty_name,
                    rbds.radio_text,
                    rbds.pty,
                    rbds.tp,
                    rbds.ta,
                    rbds.ms,
                    rbds.clock_time_local,
                )
                self._rbds_data = rbds
                # Write all fields unconditionally so cleared values propagate
                self.metrics.metadata['rbds_ps_name'] = rbds.ps_name
                self.metrics.metadata['rbds_pi_code'] = rbds.pi_code
                self.metrics.metadata['rbds_call_sign'] = rbds.call_sign
                self.metrics.metadata['rbds_pty_name'] = rbds.pty_name
                self.metrics.metadata['rbds_radio_text'] = rbds.radio_text
                self.metrics.metadata['rbds_pty'] = rbds.pty
                self.metrics.metadata['rbds_program_type_name'] = (
                    RBDS_PROGRAM_TYPES.get(int(rbds.pty), f"Unknown ({rbds.pty})")
                    if rbds.pty is not None else None
                )
                self.metrics.metadata['rbds_tp'] = rbds.tp
                self.metrics.metadata['rbds_ta'] = rbds.ta
                self.metrics.metadata['rbds_ms'] = rbds.ms
                self.metrics.metadata['rbds_di_stereo'] = rbds.di_stereo
                self.metrics.metadata['rbds_di_artificial_head'] = rbds.di_artificial_head
                self.metrics.metadata['rbds_di_compressed'] = rbds.di_compressed
                self.metrics.metadata['rbds_di_dynamic_pty'] = rbds.di_dynamic_pty
                self.metrics.metadata['rbds_clock_time_utc'] = rbds.clock_time_utc
                self.metrics.metadata['rbds_clock_time_local'] = rbds.clock_time_local
                self.metrics.metadata['rbds_af_list'] = rbds.af_list
                self.metrics.metadata['rbds_pin_day'] = rbds.pin_day
                self.metrics.metadata['rbds_pin_hour'] = rbds.pin_hour
                self.metrics.metadata['rbds_pin_minute'] = rbds.pin_minute
                self.metrics.metadata['rbds_ecc'] = rbds.ecc
                self.metrics.metadata['rbds_language_code'] = rbds.language_code
                self.metrics.metadata['rbds_language_name'] = rbds.language_name
                self.metrics.metadata['rbds_linkage_set_number'] = rbds.linkage_set_number
                self.metrics.metadata['rbds_linkage_actuator'] = rbds.linkage_actuator
                self.metrics.metadata['rbds_linkage_soft_coupling'] = rbds.linkage_soft_coupling
                self.metrics.metadata['rbds_oda_apps'] = rbds.oda_apps
                self.metrics.metadata['rbds_tdc_data'] = (
                    rbds.tdc_data.hex() if rbds.tdc_data else None
                )
                self.metrics.metadata['rbds_in_house_data'] = rbds.in_house_data
                self.metrics.metadata['rbds_tmc_present'] = rbds.tmc_present
                self.metrics.metadata['rbds_ews_channel'] = rbds.ews_channel
                self.metrics.metadata['rbds_ews_message_c'] = rbds.ews_message_c
                self.metrics.metadata['rbds_ews_message_d'] = rbds.ews_message_d
                self.metrics.metadata['rbds_eon_list'] = rbds.eon_list
                self.metrics.metadata['rbds_fast_tp'] = rbds.fast_tp
                self.metrics.metadata['rbds_fast_ta'] = rbds.fast_ta
                self.metrics.metadata['rbds_fast_ms'] = rbds.fast_ms
                self.metrics.metadata['rbds_fast_di_bits'] = rbds.fast_di_bits
                self.metrics.metadata['rbds_rt_plus_item_running'] = rbds.rt_plus_item_running
                self.metrics.metadata['rbds_rt_plus_item_toggle'] = rbds.rt_plus_item_toggle
                self.metrics.metadata['rbds_rt_plus_tags'] = rbds.rt_plus_tags
                self.metrics.metadata['rbds_radio_text_ab'] = rbds.radio_text_ab
                self.metrics.metadata['rbds_pi_country_code'] = rbds.pi_country_code
                self.metrics.metadata['rbds_pi_area_code'] = rbds.pi_area_code
                self.metrics.metadata['rbds_pi_program_ref'] = rbds.pi_program_ref
                self.metrics.metadata['rbds_oda_assignments'] = rbds.oda_assignments
                self.metrics.metadata['rbds_oda_payloads'] = rbds.oda_payloads
                self.metrics.metadata['rbds_af_method_a_count'] = rbds.af_method_a_count
                self.metrics.metadata['rbds_af_follow_on_indicator'] = rbds.af_follow_on_indicator
                self.metrics.metadata['rbds_af_method_b'] = rbds.af_method_b
                self.metrics.metadata['rbds_af_tuning_frequency'] = rbds.af_tuning_frequency
                self.metrics.metadata['rbds_paging_messages'] = rbds.paging_messages
                self.metrics.metadata['rbds_enhanced_paging_messages'] = rbds.enhanced_paging_messages
                self.metrics.metadata['rbds_paging_tmc_id'] = rbds.paging_tmc_id
                self.metrics.metadata['rbds_paging_operator_code'] = rbds.paging_operator_code
                self.metrics.metadata['rbds_ews_channel_identifier'] = rbds.ews_channel_identifier
                self.metrics.metadata['rbds_slow_labelling_raw'] = (
                    {str(k): v for k, v in rbds.slow_labelling_raw.items()}
                    if rbds.slow_labelling_raw else None
                )
                self.metrics.metadata['rbds_tdc_channels'] = (
                    {str(ch): buf.hex() for ch, buf in rbds.tdc_channels.items()}
                    if rbds.tdc_channels else None
                )
                # rbds_last_seen advances every time we observe a decoded
                # group, even if the content is identical to the last one.
                # This is the "decoder is alive" heartbeat: it lets the
                # UI distinguish "station is just playing stable content"
                # from "sync died and we're showing stale data".
                now = time.time()
                self.metrics.metadata['rbds_last_seen'] = now
                # rbds_last_updated only advances when decoded content
                # actually changes — this is the "content freshness" time
                # the user cares about for RT / PS rotation.
                if signature != self._rbds_signature:
                    self.metrics.metadata['rbds_last_updated'] = now
                    self._rbds_signature = signature
                    logger.debug(
                        f"RBDS data updated: PS={rbds.ps_name}, "
                        f"PI={rbds.pi_code} ({rbds.call_sign}), PTY={rbds.pty}"
                    )
            elif self._rbds_data is not None:
                # No new decode this cycle — keep last-known values on display
                last = self._rbds_data
                self.metrics.metadata['rbds_ps_name'] = last.ps_name
                self.metrics.metadata['rbds_pi_code'] = last.pi_code
                self.metrics.metadata['rbds_call_sign'] = last.call_sign
                self.metrics.metadata['rbds_pty_name'] = last.pty_name
                self.metrics.metadata['rbds_radio_text'] = last.radio_text
                self.metrics.metadata['rbds_pty'] = last.pty
                self.metrics.metadata['rbds_program_type_name'] = (
                    RBDS_PROGRAM_TYPES.get(int(last.pty), f"Unknown ({last.pty})")
                    if last.pty is not None else None
                )
                self.metrics.metadata['rbds_tp'] = last.tp
                self.metrics.metadata['rbds_ta'] = last.ta
                self.metrics.metadata['rbds_ms'] = last.ms
                self.metrics.metadata['rbds_di_stereo'] = last.di_stereo
                self.metrics.metadata['rbds_di_artificial_head'] = last.di_artificial_head
                self.metrics.metadata['rbds_di_compressed'] = last.di_compressed
                self.metrics.metadata['rbds_di_dynamic_pty'] = last.di_dynamic_pty
                self.metrics.metadata['rbds_clock_time_utc'] = last.clock_time_utc
                self.metrics.metadata['rbds_clock_time_local'] = last.clock_time_local
                self.metrics.metadata['rbds_af_list'] = last.af_list
                self.metrics.metadata['rbds_pin_day'] = last.pin_day
                self.metrics.metadata['rbds_pin_hour'] = last.pin_hour
                self.metrics.metadata['rbds_pin_minute'] = last.pin_minute
                self.metrics.metadata['rbds_ecc'] = last.ecc
                self.metrics.metadata['rbds_language_code'] = last.language_code
                self.metrics.metadata['rbds_language_name'] = last.language_name
                self.metrics.metadata['rbds_linkage_set_number'] = last.linkage_set_number
                self.metrics.metadata['rbds_linkage_actuator'] = last.linkage_actuator
                self.metrics.metadata['rbds_linkage_soft_coupling'] = last.linkage_soft_coupling
                self.metrics.metadata['rbds_oda_apps'] = last.oda_apps
                self.metrics.metadata['rbds_tdc_data'] = (
                    last.tdc_data.hex() if last.tdc_data else None
                )
                self.metrics.metadata['rbds_in_house_data'] = last.in_house_data
                self.metrics.metadata['rbds_tmc_present'] = last.tmc_present
                self.metrics.metadata['rbds_ews_channel'] = last.ews_channel
                self.metrics.metadata['rbds_ews_message_c'] = last.ews_message_c
                self.metrics.metadata['rbds_ews_message_d'] = last.ews_message_d
                self.metrics.metadata['rbds_eon_list'] = last.eon_list
                self.metrics.metadata['rbds_fast_tp'] = last.fast_tp
                self.metrics.metadata['rbds_fast_ta'] = last.fast_ta
                self.metrics.metadata['rbds_fast_ms'] = last.fast_ms
                self.metrics.metadata['rbds_fast_di_bits'] = last.fast_di_bits
                self.metrics.metadata['rbds_rt_plus_item_running'] = last.rt_plus_item_running
                self.metrics.metadata['rbds_rt_plus_item_toggle'] = last.rt_plus_item_toggle
                self.metrics.metadata['rbds_rt_plus_tags'] = last.rt_plus_tags
                self.metrics.metadata['rbds_radio_text_ab'] = last.radio_text_ab
                self.metrics.metadata['rbds_pi_country_code'] = last.pi_country_code
                self.metrics.metadata['rbds_pi_area_code'] = last.pi_area_code
                self.metrics.metadata['rbds_pi_program_ref'] = last.pi_program_ref
                self.metrics.metadata['rbds_oda_assignments'] = last.oda_assignments
                self.metrics.metadata['rbds_oda_payloads'] = last.oda_payloads
                self.metrics.metadata['rbds_af_method_a_count'] = last.af_method_a_count
                self.metrics.metadata['rbds_af_follow_on_indicator'] = last.af_follow_on_indicator
                self.metrics.metadata['rbds_af_method_b'] = last.af_method_b
                self.metrics.metadata['rbds_af_tuning_frequency'] = last.af_tuning_frequency
                self.metrics.metadata['rbds_paging_messages'] = last.paging_messages
                self.metrics.metadata['rbds_enhanced_paging_messages'] = last.enhanced_paging_messages
                self.metrics.metadata['rbds_paging_tmc_id'] = last.paging_tmc_id
                self.metrics.metadata['rbds_paging_operator_code'] = last.paging_operator_code
                self.metrics.metadata['rbds_ews_channel_identifier'] = last.ews_channel_identifier
                self.metrics.metadata['rbds_slow_labelling_raw'] = (
                    {str(k): v for k, v in last.slow_labelling_raw.items()}
                    if last.slow_labelling_raw else None
                )
                self.metrics.metadata['rbds_tdc_channels'] = (
                    {str(ch): buf.hex() for ch, buf in last.tdc_channels.items()}
                    if last.tdc_channels else None
                )
            else:
                # No cached data and nothing new this cycle — e.g. right
                # after a frequency change.  Publish explicit nulls so
                # the UI clears the previous station's PS/RT instead of
                # holding on to whatever was there before.
                self.metrics.metadata['rbds_ps_name'] = None
                self.metrics.metadata['rbds_pi_code'] = None
                self.metrics.metadata['rbds_call_sign'] = None
                self.metrics.metadata['rbds_pty_name'] = None
                self.metrics.metadata['rbds_radio_text'] = None
                self.metrics.metadata['rbds_pty'] = None
                self.metrics.metadata['rbds_program_type_name'] = None
                self.metrics.metadata['rbds_tp'] = None
                self.metrics.metadata['rbds_ta'] = None
                self.metrics.metadata['rbds_di_stereo'] = None
                self.metrics.metadata['rbds_di_artificial_head'] = None
                self.metrics.metadata['rbds_di_compressed'] = None
                self.metrics.metadata['rbds_di_dynamic_pty'] = None
                self.metrics.metadata['rbds_clock_time_utc'] = None
                self.metrics.metadata['rbds_clock_time_local'] = None
                self.metrics.metadata['rbds_ms'] = None
