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

"""Tests for the signal-quality history API
(/api/audio/sources/<name>/signal_quality/history).

The endpoint derives a stereo-pilot-strength + click-rate trend series from
the AudioSourceMetrics snapshots the audio service persists once a second --
same source table as the RBDS history endpoint (see
tests/test_rbds_history_api.py), different fields.
"""

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from flask import Flask
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_core.extensions import db
from app_core.models import AudioSourceMetrics
from app_utils import utc_now
from webapp.admin import audio_ingest as audio_admin
from webapp.admin.audio_ingest import controller as audio_controller_mod


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kwargs):  # pragma: no cover - sqlalchemy hook
    return "TEXT"


@pytest.fixture(autouse=True)
def _quiet_audio_globals(monkeypatch):
    monkeypatch.setattr(audio_controller_mod, "_initialization_started", True)
    monkeypatch.setattr(audio_controller_mod, "_start_audio_sources_background", lambda app: None)


@pytest.fixture
def history_app(tmp_path: Path):
    database_path = tmp_path / "signal_quality_history.db"
    app = Flask("signal-quality-history-test")
    app.config.update(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{database_path}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)

    with app.app_context():
        AudioSourceMetrics.__table__.create(bind=db.engine)
        audio_admin.register_audio_ingest_routes(app, logging.getLogger("signal-quality-history-test"))
        yield app
        db.session.remove()
        AudioSourceMetrics.__table__.drop(bind=db.engine)


def _add_snapshot(ts, metadata, source_name="sdr-test"):
    db.session.add(AudioSourceMetrics(
        source_name=source_name,
        source_type="sdr",
        peak_level_db=-6.0,
        rms_level_db=-18.0,
        peak_level_linear=0.5,
        rms_level_linear=0.12,
        sample_rate=48000,
        channels=2,
        frames_captured=0,
        timestamp=ts,
        source_metadata=metadata,
    ))


def test_history_returns_pilot_and_click_rate_points(history_app):
    with history_app.app_context():
        base = utc_now() - timedelta(minutes=10)
        for i in range(8):
            md = {
                'stereo_pilot_strength': 0.1 * i,
                'click_rate': 0.01 * i,
                'click_suppression_enabled': True,
            }
            _add_snapshot(base + timedelta(seconds=30 * i), md)
        db.session.commit()

        client = history_app.test_client()
        resp = client.get('/api/audio/sources/sdr-test/signal_quality/history?minutes=60')
        assert resp.status_code == 200
        data = resp.get_json()

        assert data['source'] == 'sdr-test'
        assert data['sample_count'] == 8
        assert len(data['points']) == 8
        assert data['points'][3]['stereo_pilot_strength'] == pytest.approx(0.3)
        assert data['points'][3]['click_rate'] == pytest.approx(0.03)


def test_history_does_not_leak_rf_signal_strength(history_app):
    """RF signal strength is deliberately excluded (see routes_signal_quality.py
    module docstring) -- it's already charted from a different, hardware-level
    source elsewhere on the same page. A stray rf_signal_strength key in the
    metadata must not leak into this endpoint's response shape."""
    with history_app.app_context():
        base = utc_now() - timedelta(minutes=5)
        _add_snapshot(base, {
            'stereo_pilot_strength': 0.9,
            'click_rate': 0.0,
            'rf_signal_strength': 1.23,
        })
        db.session.commit()

        client = history_app.test_client()
        resp = client.get('/api/audio/sources/sdr-test/signal_quality/history')
        assert resp.status_code == 200
        data = resp.get_json()

        assert data['points'][0]['stereo_pilot_strength'] == pytest.approx(0.9)
        assert 'rf_signal_strength' not in data['points'][0]


def test_history_skips_snapshots_with_neither_field(history_app):
    with history_app.app_context():
        base = utc_now() - timedelta(minutes=5)
        _add_snapshot(base, {'stereo_pilot_strength': 0.5})
        # No pilot/click-rate keys at all — not a signal-quality sample, skipped.
        _add_snapshot(base + timedelta(seconds=30), {'rbds_lock_state': 'LOCKED'})
        _add_snapshot(base + timedelta(seconds=60), {'click_rate': 0.02})
        db.session.commit()

        client = history_app.test_client()
        resp = client.get('/api/audio/sources/sdr-test/signal_quality/history')
        assert resp.status_code == 200
        data = resp.get_json()

        assert data['sample_count'] == 2


def test_history_unknown_source_and_bad_minutes(history_app):
    with history_app.app_context():
        client = history_app.test_client()
        resp = client.get('/api/audio/sources/no-such-source/signal_quality/history?minutes=banana')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['minutes'] == 60  # malformed value falls back to default
        assert data['points'] == []


def test_history_downsamples_long_series(history_app):
    with history_app.app_context():
        base = utc_now() - timedelta(minutes=30)
        for i in range(1000):
            _add_snapshot(
                base + timedelta(seconds=i),
                {'stereo_pilot_strength': 0.5, 'click_rate': 0.0},
            )
        db.session.commit()

        client = history_app.test_client()
        resp = client.get('/api/audio/sources/sdr-test/signal_quality/history?minutes=60')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['sample_count'] == 1000
        assert len(data['points']) <= 500
