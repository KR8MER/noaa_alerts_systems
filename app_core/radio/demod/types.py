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

"""Dataclasses describing demodulator configuration, status and RBDS output."""

import base64
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DemodulatorConfig:
    """Configuration for audio demodulator."""
    modulation_type: str  # 'FM', 'WFM', 'NFM', 'AM', 'IQ'
    sample_rate: int  # Input sample rate (Hz)
    audio_sample_rate: int = 44100  # Output audio sample rate (native for streams/outputs)
    stereo_enabled: bool = True  # Enable FM stereo decoding
    deemphasis_us: float = 75.0  # De-emphasis time constant (75μs NA, 50μs EU, 0 to disable)
    enable_rbds: bool = False  # Extract RBDS data from FM multiplex

    # Magnitude-aware FM click suppression — adapts the discriminator to
    # imperfect signals (multipath fades from nearby strong stations, urban
    # propagation, etc.).  When the IQ envelope momentarily collapses, the
    # raw atan2 output is uniformly-distributed phase noise that spreads as
    # a flat impulse floor across the entire MPX band and buries the
    # 19 kHz pilot / 57 kHz RBDS lobes.  The suppressor detects those
    # samples by their magnitude and forward-fills the previous good
    # phase.  Default ON because the math is conservative (only triggers
    # well below normal modulation depth) and even clean signals benefit
    # from quieter noise around the subcarriers.
    enable_click_suppression: bool = True
    # Fraction of mean |z|^2 below which a sample is treated as a click.
    # 0.0 disables.  0.1 is the value validated against the 7-mile Class B
    # capture (bugs/iq_sdr_256000Hz_1778771016_f69ab915.npz): drops click
    # rate from 9.5 % to <1 % without disturbing legitimate modulation
    # peaks.  Higher values are more aggressive; values above ~0.4 will
    # start suppressing legitimate signal and should not be used in
    # production.
    click_suppression_threshold: float = 0.1


@dataclass
class RBDSData:
    """Decoded RBDS/RDS data from FM broadcast."""
    pi_code: Optional[str] = None  # Program Identification (raw 16-bit hex)
    # PI is structured as 4 bits country code + 4 bits area/coverage code +
    # 8 bits programme reference (Annex D of NRSC-4-B).  For US stations
    # this is a synthetic-call-sign mapping; for European stations the
    # split is the actual country/region/programme identifier.
    pi_country_code: Optional[int] = None       # high 4 bits
    pi_area_code: Optional[int] = None          # next 4 bits (coverage area)
    pi_program_ref: Optional[int] = None        # low 8 bits
    call_sign: Optional[str] = None  # Decoded US call letters (e.g. "WXYZ"), if PI is US
    ps_name: Optional[str] = None  # Program Service name (8 chars)
    pty_name: Optional[str] = None  # Program Type Name (PTYN, 8 chars, Group 10A)
    radio_text: Optional[str] = None  # Radio Text (up to 64 chars)
    # RT A/B flag.  The station toggles this every time the displayed
    # message restarts; the UI can use it to detect when an RT change is
    # in progress (vs. just being extended segment-by-segment).
    radio_text_ab: Optional[int] = None
    pty: Optional[int] = None  # Program Type
    tp: Optional[bool] = None  # Traffic Program flag
    ta: Optional[bool] = None  # Traffic Announcement flag
    ms: Optional[bool] = None  # Music/Speech flag
    di_stereo: Optional[bool] = None  # Decoder Identification: stereo
    di_artificial_head: Optional[bool] = None  # Decoder Identification: artificial head
    di_compressed: Optional[bool] = None  # Decoder Identification: compressed audio
    di_dynamic_pty: Optional[bool] = None  # Decoder Identification: dynamic PTY
    clock_time_utc: Optional[str] = None  # ISO-8601 UTC timestamp from Group 4A
    clock_time_local: Optional[str] = None  # ISO-8601 local timestamp from Group 4A
    # Group 0A Block C - Alternative Frequencies
    af_list: Optional[List[float]] = None  # list of AF frequencies in MHz
    # Method-A AF list count (codes 224-249).  None until announced; once
    # set, len(af_list) == af_method_a_count means the list is complete.
    af_method_a_count: Optional[int] = None
    # True if the station has signalled an LF/MF follow-on (code 250),
    # i.e. the AF list extends to non-VHF frequencies we can't represent
    # in MHz directly.  Pure indicator — no further data captured.
    af_follow_on_indicator: Optional[bool] = None
    # True once the station has emitted a Method-B regional-variant
    # marker (an AF pair where both bytes are the same direct code).
    # Method B and Method A both still produce frequencies in af_list;
    # this flag plus af_tuning_frequency just say "this station also
    # broadcasts the AF list paired against the tuned frequency".
    af_method_b: Optional[bool] = None
    af_tuning_frequency: Optional[float] = None
    # Group 1A/1B - Programme Item Number + Slow Labeling Codes
    pin_day: Optional[int] = None
    pin_hour: Optional[int] = None
    pin_minute: Optional[int] = None
    ecc: Optional[int] = None  # Extended Country Code
    language_code: Optional[int] = None
    language_name: Optional[str] = None
    linkage_set_number: Optional[int] = None
    linkage_actuator: Optional[bool] = None
    linkage_soft_coupling: Optional[bool] = None
    # Group 1A variant-1: 12-bit TMC identification provider code.
    paging_tmc_id: Optional[int] = None
    # Group 1A variant-2: 12-bit paging operator code.
    paging_operator_code: Optional[int] = None
    # Group 1A variant-7: 12-bit EWS slow-labelling channel identifier
    # (which EWS provider feeds the Group 9A messages on this station).
    ews_channel_identifier: Optional[int] = None
    # Raw 16-bit Block-D payload per Group 1A variant (0..7), so the UI
    # can show every variant byte the station broadcasts even where no
    # decoder semantics apply.
    slow_labelling_raw: Optional[Dict[int, int]] = None
    # Group 3A - Open Data Application registration
    oda_apps: Optional[List[int]] = None
    # Full ODA assignment table — list of {group, version, aid, aid_hex,
    # name?} items so the UI can show *which* group type carries each
    # registered application instead of just listing AIDs.
    oda_assignments: Optional[List[dict]] = None
    # Per-AID raw payload buffer for ODAs we don't have a specific
    # decoder for.  Each entry: {aid, aid_hex, group, count,
    # last_b_low, last_c, last_d, last_seen_unix}.
    oda_payloads: Optional[List[dict]] = None
    # Group 7A / 13A paging messages.  Each is a list of dicts with
    # the raw bytes and a unix timestamp; format is operator-defined,
    # so the decoder doesn't try to interpret the payload.
    paging_messages: Optional[List[dict]] = None
    enhanced_paging_messages: Optional[List[dict]] = None
    # Group 5A/5B - Transparent Data Channel
    # tdc_data is channel 0 (kept for backwards compat); tdc_channels
    # exposes every channel TS the station broadcasts.
    tdc_data: Optional[bytes] = None
    tdc_channels: Optional[Dict[int, bytes]] = None
    # Group 6A/6B - In-House Applications
    in_house_data: Optional[List[int]] = None
    # Group 8A - Traffic Message Channel
    tmc_present: Optional[bool] = None
    # Group 9A - Emergency Warning System
    ews_channel: Optional[int] = None
    ews_message_c: Optional[int] = None
    ews_message_d: Optional[int] = None
    # Group 14A/14B - Enhanced Other Networks
    eon_list: Optional[List[dict]] = None
    # Group 15B - Fast Switching Information
    fast_tp: Optional[bool] = None
    fast_ta: Optional[bool] = None
    fast_ms: Optional[bool] = None
    fast_di_bits: Optional[int] = None
    # RT+ (ODA AID 0xCD46) - structured tags pointing into Radio Text.
    # Each tag is a dict {content_type, content_type_name, text, start, length}.
    rt_plus_item_running: Optional[bool] = None
    rt_plus_item_toggle: Optional[int] = None
    rt_plus_tags: Optional[List[dict]] = None


@dataclass
class RBDSDecoderStats:
    """Snapshot of RBDS decoder health and traffic.

    Reset on every sync acquisition (so values reflect the *current* lock
    rather than lifetime totals across station changes), except for
    ``sync_lost_count`` which is cumulative since the worker was started.
    Surfaces what redsea calls "block error rate" plus a per-group-type
    traffic histogram so operators can tell *why* a station is missing
    metadata (e.g. it doesn't broadcast any 2A groups, so RT will never
    appear).
    """
    blocks_total: int = 0
    blocks_ok: int = 0          # passed CRC without FEC
    blocks_fec_single: int = 0  # repaired by single-bit corrector
    blocks_fec_burst: int = 0   # repaired by burst-trapping decoder
    blocks_uncorrected: int = 0
    # Blocks recovered by ±1-bit clock-slip realignment: the block grid was
    # shifted one bit by an M&M symbol-timing slip and the decoder realigned
    # in place instead of dropping sync.  These blocks also count in
    # blocks_ok (they passed CRC cleanly once realigned); a high rate here
    # with low BLER points at symbol-timing trouble, not RF noise.
    blocks_bit_slips: int = 0
    groups_decoded: int = 0
    sync_acquired_unix: Optional[float] = None
    sync_lost_count: int = 0
    # Number of sample chunks dropped because the worker queue was full.
    # A non-zero value here means the DSP thread is behind real-time;
    # it triggers M&M timing phase errors because state carries over
    # across the resulting time gaps.
    chunks_dropped: int = 0
    # Keys are e.g. "0A", "2A", "11A" — bare "A"/"B" suffixes match what
    # the RDS specs use everywhere so the UI doesn't have to translate.
    group_type_counts: Dict[str, int] = field(default_factory=dict)
    # Field-level churn counters from the two-sighting confirmation gate.
    # pi/pty/ta count *accepted* value changes after the field first
    # resolved — on a station that stays tuned these should sit at 0
    # (a mid-lock PI change means either an actual station change or a
    # repeated glitch that beat the voting gate).  glitches_rejected
    # counts single-sighting candidates that were contradicted by the
    # next observation before they could confirm — i.e. false reads the
    # gate stopped from reaching the UI.  Unlike the block counters
    # these are cumulative since tune/reset, not per-lock, because the
    # decoder's field state also survives a sync drop.
    pi_change_count: int = 0
    pty_change_count: int = 0
    ta_toggle_count: int = 0
    glitches_rejected: int = 0

    @property
    def raw_block_error_rate(self) -> Optional[float]:
        """Fraction of received blocks that didn't pass CRC on first try.

        This is the NRSC-4-B §7.4.2 definition of BLER — any block whose
        syndrome was non-zero before FEC counts as an error, regardless
        of whether FEC later repaired it.  Returns None until at least
        one block has been processed (so the UI doesn't display 0/0).
        """
        if self.blocks_total == 0:
            return None
        return (self.blocks_total - self.blocks_ok) / self.blocks_total

    @property
    def net_block_error_rate(self) -> Optional[float]:
        """Fraction of blocks the decoder still couldn't recover after FEC.

        Operationally what users care about — this is what drives PS/RT
        gaps.  raw - net = "how much FEC saved us".
        """
        if self.blocks_total == 0:
            return None
        return self.blocks_uncorrected / self.blocks_total


@dataclass
class DemodulatorStatus:
    """Status information from FM demodulator."""
    rbds_data: Optional[RBDSData] = None  # RBDS data if available
    stereo_pilot_locked: bool = False  # 19 kHz stereo pilot detected
    stereo_pilot_strength: float = 0.0  # Pilot signal strength (0.0 to 1.0)
    is_stereo: bool = False  # Stereo decoding active
    # Mean magnitude of the IQ samples for this chunk (linear, 0.0-1.0 range
    # for normalized float samples).  The UI converts this to dBFS for the
    # RF RSSI meter.
    signal_strength: float = 0.0
    # True once the RBDS bit-level sync state machine has locked.  Lets the
    # UI show a "LOCKING" vs "LOCKED" indicator instead of leaving users
    # guessing why no data has appeared yet.
    rbds_synced: bool = False
    # True if the demodulator is even attempting RBDS decoding (i.e. the
    # receiver has enable_rbds set and the IQ sample rate is high enough
    # to preserve the 57 kHz subcarrier).  Without this flag a receiver
    # configured with enable_rbds=False looked indistinguishable from one
    # that was still acquiring sync — both produced rbds_synced=False —
    # and users could wait forever for a lock the decoder never even
    # tried to obtain.
    rbds_enabled: bool = False
    # Decoder-side health metrics — block error rate, FEC correction
    # split, and group-type histogram.  None until an RBDS worker is
    # running; otherwise updated each frame.
    rbds_decoder_stats: Optional[RBDSDecoderStats] = None
    # Fraction (0.0–1.0) of discriminator output samples in this chunk
    # that the magnitude-aware click suppressor replaced.  High values
    # (>0.05) indicate impulse-noise-limited reception — typically deep
    # multipath fades on otherwise strong signals — and explain RBDS
    # decoder stalls that aren't visible in the simple RSSI meter.
    # 0.0 when suppression is disabled or no samples were suppressed.
    click_rate: float = 0.0
    # True if the magnitude-aware click suppressor is active on this
    # demodulator.  Lets the UI distinguish "0% clicks because suppressor
    # is off" from "0% clicks because the signal is clean".
    click_suppression_enabled: bool = False
    # Peak instantaneous deviation of the whole composite MPX signal this
    # chunk, in Hz -- the FCC full-scale reference is +/-75 kHz. Derived
    # from the same raw discriminator output (radians/sample) already
    # computed for every chunk regardless; not a new DSP stage, just the
    # inverse of the audio_gain scale factor already used to normalize
    # audio (see WFMDemodulator._audio_gain).
    peak_deviation_hz: float = 0.0
    # RMS injection level of the 19 kHz stereo pilot, in Hz. A healthy FM
    # stereo broadcast targets ~6.75 kHz (9% of 75 kHz full-scale). Reuses
    # the same pilot_rms value already computed for stereo_pilot_strength
    # (which stays an arbitrary 0-1 UI/lock-threshold scale) -- this is
    # the calibrated Hz reading a broadcast engineer actually wants.
    pilot_injection_hz: float = 0.0
    # RMS injection level of the 57 kHz RDS subcarrier, in Hz (typical
    # healthy range ~2-4.5 kHz). Only populated when RBDS decoding is
    # enabled and the sample rate preserves the subcarrier (same gate as
    # rbds_enabled) -- unlike pilot injection, this needs a small new
    # bandpass filter (mirrors the existing pilot filter exactly).
    rds_injection_hz: float = 0.0


# ── JSON round-trip for cross-process status sharing ────────────────────────
# The demod worker process publishes DemodulatorStatus to Redis so the web
# process's SDR adapter can read it back (see services/demod/worker.py and
# app_core/audio/redis_sdr_adapter.py). This used to go through
# pickle.dumps/pickle.loads, which is an arbitrary-code-execution primitive
# the moment anything untrusted can write to that Redis key -- not exploitable
# today (only this app's own worker writes it), but a latent risk not worth
# keeping when the actual data is a plain dataclass tree. JSON can't
# represent `bytes` or int-typed dict keys directly (RBDSData.tdc_data,
# tdc_channels, slow_labelling_raw), so those are base64/str-key encoded
# going out and decoded going back in -- everything else round-trips through
# plain dataclasses.asdict()/the constructor unchanged.
def demodulator_status_to_json_dict(status: 'DemodulatorStatus') -> dict:
    """Convert a DemodulatorStatus into a plain, json.dumps-safe dict."""
    data = dataclasses.asdict(status)
    rbds_data = data.get('rbds_data')
    if rbds_data is not None:
        if rbds_data.get('tdc_data') is not None:
            rbds_data['tdc_data'] = base64.b64encode(rbds_data['tdc_data']).decode('ascii')
        if rbds_data.get('tdc_channels') is not None:
            rbds_data['tdc_channels'] = {
                str(k): base64.b64encode(v).decode('ascii')
                for k, v in rbds_data['tdc_channels'].items()
            }
        if rbds_data.get('slow_labelling_raw') is not None:
            rbds_data['slow_labelling_raw'] = {
                str(k): v for k, v in rbds_data['slow_labelling_raw'].items()
            }
    return data


def demodulator_status_from_json_dict(data: Dict[str, Any]) -> 'DemodulatorStatus':
    """Reconstruct a DemodulatorStatus from demodulator_status_to_json_dict's output."""
    data = dict(data)
    rbds_data = data.get('rbds_data')
    if rbds_data is not None:
        rbds_data = dict(rbds_data)
        if rbds_data.get('tdc_data') is not None:
            rbds_data['tdc_data'] = base64.b64decode(rbds_data['tdc_data'])
        if rbds_data.get('tdc_channels') is not None:
            rbds_data['tdc_channels'] = {
                int(k): base64.b64decode(v) for k, v in rbds_data['tdc_channels'].items()
            }
        if rbds_data.get('slow_labelling_raw') is not None:
            rbds_data['slow_labelling_raw'] = {
                int(k): v for k, v in rbds_data['slow_labelling_raw'].items()
            }
        data['rbds_data'] = RBDSData(**rbds_data)
    rbds_decoder_stats = data.get('rbds_decoder_stats')
    if rbds_decoder_stats is not None:
        data['rbds_decoder_stats'] = RBDSDecoderStats(**rbds_decoder_stats)
    return DemodulatorStatus(**data)

