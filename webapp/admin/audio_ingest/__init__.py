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

"""Audio Ingest API routes for managing audio sources and monitoring.

This package was a single 3,180-line ``webapp/admin/audio_ingest.py``. It is
split by concern now, and this module is the compatibility shim: everything the
old module exposed is re-exported here, so ``from webapp.admin.audio_ingest
import …`` keeps resolving for ``webapp/routes_settings_radio.py``,
``app_core/websocket_push.py`` and the tests.

Layout:

    blueprint              the shared ``audio_ingest_bp``
    controller             controller singleton, startup, Redis metrics bridge
    streaming              auto-streaming (Icecast) service lifecycle
    sanitize               JSON-safety helpers
    probe                  stream-URL probing
    radio_sources          SDR-backed source provisioning
    serialization          source -> API payload
    routes_*               the endpoints, grouped by resource

**Patching module state from a test targets the submodule, not this package.**
Rebinding a name here does not change what a submodule sees in its own globals,
so ``monkeypatch.setattr(audio_ingest.controller, "_audio_controller", …)`` is
the form that works — see ``tests/test_radio_audio_monitoring.py``.
"""

import logging
from typing import Any

from flask import Flask

from . import (  # noqa: F401  - imported for their side effect of registering routes
    blueprint,
    controller,
    probe,
    radio_sources,
    listing,
    routes_alerts,
    routes_dead_air,
    routes_devices,
    routes_health,
    routes_icecast,
    routes_metrics,
    routes_rbds,
    routes_signal_quality,
    routes_source_control,
    routes_sources,
    routes_sources_delete,
    routes_sources_write,
    sanitize,
    source_payload,
    serialization,
    streaming,
)
from .blueprint import audio_ingest_bp
from .controller import (
    _get_audio_controller,
    _load_audio_source_configs,
    _peek_audio_controller,
    _read_audio_metrics_from_redis,
    _start_audio_sources_background,
    _try_acquire_lock,
)
from .probe import _describe_stream_status, _probe_stream_url
from .radio_sources import (
    _base_radio_metadata,
    _derive_sdr_source_name,
    _format_receiver_frequency,
    _recommend_audio_stream,
    ensure_sdr_audio_monitor_source,
    list_radio_managed_audio_sources,
    remove_radio_managed_audio_source,
)
from .routes_alerts import (
    api_acknowledge_alert,
    api_get_audio_alerts,
    api_resolve_alert,
    api_resolve_all_alerts,
)
from .routes_devices import (
    api_discover_audio_devices,
    api_get_spectrogram,
    api_get_waveform,
    api_stream_audio,
)
from .routes_health import (
    api_get_audio_health,
    api_get_health_dashboard,
    api_get_health_metrics,
    audio_health_dashboard,
)
from .routes_icecast import (
    api_get_icecast_config,
    api_start_icecast_stream,
    api_stop_icecast_stream,
    api_update_icecast_config,
)
from .routes_metrics import api_get_audio_metrics, api_get_audio_metrics_latest
from .routes_rbds import (
    _RBDS_EVENT_FIELDS,
    _RBDS_HISTORY_MAX_EVENTS,
    _RBDS_HISTORY_MAX_POINTS,
    api_get_rbds_history,
)
from .routes_signal_quality import (
    _SIGNAL_QUALITY_HISTORY_MAX_POINTS,
    api_get_signal_quality_history,
)
from .routes_source_control import (
    api_start_audio_source,
    api_stop_audio_source,
    api_test_stream_url,
)
from .routes_sources import api_get_audio_source, api_get_audio_sources
from .routes_sources_delete import api_delete_audio_source
from .routes_sources_write import (
    api_create_audio_source,
    api_update_audio_source,
)
from .sanitize import (
    _db_to_linear,
    _merge_metadata,
    _redact_device_params,
    _sanitize_bool,
    _sanitize_float,
    _sanitize_metadata_value,
)
from .serialization import (
    _get_controller_and_adapter,
    _read_redis_source_data,
    _restore_audio_source_from_db_config,
    _sanitize_streaming_stats,
    _serialize_audio_source,
    _serialize_audio_source_from_db,
)
from .streaming import (
    _get_auto_streaming_service,
    _get_icecast_stream_url,
    _initialize_auto_streaming,
    _reload_auto_streaming_from_env,
    _safe_auto_stream_status,
    _start_auto_streaming_service,
    _stop_auto_streaming_service,
)

logger = logging.getLogger(__name__)

# Every module that carries a ``logger`` global the registration hook rebinds.
# The pre-split module had exactly one such global and
# ``register_audio_ingest_routes`` reassigned it; the fan-out below preserves
# that, since rebinding ``logger`` on this package alone would leave every
# submodule still logging under its own name.
_LOGGING_MODULES = (
    controller,
    listing,
    source_payload,
    streaming,
    sanitize,
    probe,
    radio_sources,
    serialization,
    routes_sources,
    routes_sources_delete,
    routes_sources_write,
    routes_source_control,
    routes_rbds,
    routes_signal_quality,
    routes_metrics,
    routes_health,
    listing,
    routes_alerts,
    routes_dead_air,
    routes_devices,
    routes_icecast,
)


def register_audio_ingest_routes(app: Flask, logger_instance: Any) -> None:
    """Register audio ingest API routes."""
    global logger
    logger = logger_instance
    for module in _LOGGING_MODULES:
        module.logger = logger_instance

    # Register the blueprint with the app
    app.register_blueprint(audio_ingest_bp)
    logger_instance.info("Audio ingest routes registered")


__all__ = ['register_audio_ingest_routes']
