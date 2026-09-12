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

"""Signal-quality history endpoint (stereo pilot strength, click rate).

Sibling of routes_rbds.py -- same query/downsample shape against the
``audio_source_metrics`` table, different fields. Kept separate from
routes_rbds.py because that module's ``_RBDS_EVENT_FIELDS`` and event-log
concept are specific to discrete RDS field changes (PS, RadioText, lock
state, ...), whereas these two metrics are continuous numeric series with
no analogous "event" to log.

Deliberately does NOT include RF signal strength: the page's existing
"Historical Trends" panel (static/js/radio_diagnostics_trends.js, backed by
app_core/radio/trends.py) already charts a hardware-level Signal Strength
series sampled from ReceiverStatus.signal_strength every 10s. This
endpoint's fields come from the demodulator's DemodulatorStatus instead
(stereo pilot lock/strength, discriminator click rate) -- metrics that
exist nowhere else on this page. Surfacing a second, differently-sourced
"signal strength" series next to the existing one would be redundant and
would invite "why do these two not match" confusion.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List
from flask import jsonify, request
from app_core.extensions import db
from app_core.models import (
    AudioSourceMetrics,
)
from app_utils import utc_now

from .blueprint import audio_ingest_bp

logger = logging.getLogger(__name__)


# Keep the series small enough to chart comfortably; snapshots are written
# about once a second, so an hour is ~3600 raw rows. Matches
# routes_rbds.py's _RBDS_HISTORY_MAX_POINTS so both charts on the same page
# downsample to comparable resolution.
_SIGNAL_QUALITY_HISTORY_MAX_POINTS = 480


@audio_ingest_bp.route('/api/audio/sources/<path:source_name>/signal_quality/history', methods=['GET'])
def api_get_signal_quality_history(source_name: str):
    """Historical signal-quality telemetry for one audio source.

    Derives a downsampled trend series from the ``audio_source_metrics``
    snapshots the audio service already persists once/sec (see
    ``eas_monitoring_service.py::_snapshot_audio_metrics_once``): stereo
    pilot lock strength and discriminator click rate (multipath/
    impulse-noise indicator).

    Query:
        minutes (int, optional): History window in minutes. Default 60,
            clamped to 5..1440.

    Returns:
        200 with {source, minutes, sample_count, points}, where each point
        is {t, stereo_pilot_strength, click_rate} (either field may be
        null if that snapshot didn't carry it).
    """
    try:
        minutes = int(request.args.get('minutes', 60))
    except (TypeError, ValueError):
        minutes = 60
    minutes = max(5, min(minutes, 1440))
    cutoff = utc_now() - timedelta(minutes=minutes)

    try:
        rows = (
            db.session.query(
                AudioSourceMetrics.timestamp,
                AudioSourceMetrics.source_metadata,
            )
            .filter(
                AudioSourceMetrics.source_name == source_name,
                AudioSourceMetrics.timestamp >= cutoff,
            )
            .order_by(AudioSourceMetrics.timestamp.asc())
            .limit(90000)
            .all()
        )
    except Exception as exc:
        logger.error('Error querying signal-quality history for %s: %s', source_name, exc)
        return jsonify({'error': str(exc)}), 500

    points: List[Dict[str, Any]] = []
    for ts, md in rows:
        md = md or {}
        if 'stereo_pilot_strength' not in md and 'click_rate' not in md:
            continue

        points.append({
            't': ts.isoformat() if ts is not None else None,
            'stereo_pilot_strength': md.get('stereo_pilot_strength'),
            'click_rate': md.get('click_rate'),
        })

    sample_count = len(points)
    if sample_count > _SIGNAL_QUALITY_HISTORY_MAX_POINTS:
        stride = -(-sample_count // _SIGNAL_QUALITY_HISTORY_MAX_POINTS)  # ceil div
        points = points[::stride]

    return jsonify({
        'source': source_name,
        'minutes': minutes,
        'sample_count': sample_count,
        'points': points,
    })
