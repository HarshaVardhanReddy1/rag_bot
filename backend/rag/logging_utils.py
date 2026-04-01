import json
import logging
from datetime import datetime, timezone
from functools import wraps
from time import perf_counter


rag_logger = logging.getLogger("backend.rag.pipeline")


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _build_log_payload(function_name: str, start_time: str, end_time: str, duration_ms: float, status: str, error: str | None = None):
    payload = {
        "event": "rag_pipeline_step",
        "function": function_name,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": round(duration_ms, 2),
    }

    if error:
        payload["error"] = error

    return payload


def _log_step(function_name: str, start_time: str, end_time: str, duration_ms: float, status: str, error: str | None = None):
    payload = _build_log_payload(
        function_name=function_name,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        status=status,
        error=error,
    )

    if status == "exception":
        rag_logger.exception(json.dumps(payload))
        return

    rag_logger.info(json.dumps(payload))


def log_execution_event(function_name: str, start_time: str, end_time: str, duration_ms: float, status: str, error: str | None = None):
    _log_step(
        function_name=function_name,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        status=status,
        error=error,
    )


def log_execution_time(function_name: str | None = None):
    def decorator(func):
        step_name = function_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = _utc_timestamp()
            start_counter = perf_counter()

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                end_time = _utc_timestamp()
                duration_ms = (perf_counter() - start_counter) * 1000
                _log_step(
                    function_name=step_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    status="exception",
                    error=str(exc),
                )
                raise

            end_time = _utc_timestamp()
            duration_ms = (perf_counter() - start_counter) * 1000
            _log_step(
                function_name=step_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                status="success",
            )
            return result

        return wrapper

    return decorator
