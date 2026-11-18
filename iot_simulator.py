"""Virtual IoT smart-lock controller for the P300 authentication project."""

from datetime import datetime, timezone
from threading import RLock


class VirtualIoTController:
    """Software simulation of an ESP32-controlled servo smart lock."""

    def __init__(self):
        self._lock = RLock()
        self._device_id = "VIRTUAL-ESP32-01"
        self._online = True
        self._door_status = "LOCKED"
        self._servo_angle = 0
        self._last_command = "SYSTEM_INITIALIZED"
        self._last_event = None
        self._history = []

    def _snapshot(self):
        return {
            "device_id": self._device_id,
            "device_type": "Virtual ESP32",
            "online": self._online,
            "door_status": self._door_status,
            "servo_angle": self._servo_angle,
            "last_command": self._last_command,
            "last_event": self._last_event,
            "history": list(self._history),
            "simulation": True,
        }

    def get_status(self):
        with self._lock:
            return self._snapshot()

    def apply_authentication_result(
        self,
        decision,
        subject,
        source_subject,
        recording,
        window_index,
        score,
        threshold,
    ):
        """Apply a backend-generated CNN decision to the virtual lock."""

        if decision not in {"ACCEPT", "REJECT"}:
            raise ValueError("Decision must be ACCEPT or REJECT.")

        with self._lock:
            if not self._online:
                raise RuntimeError("Virtual IoT device is offline.")

            if decision == "ACCEPT":
                self._door_status = "UNLOCKED"
                self._servo_angle = 90
                command = "UNLOCK"
            else:
                self._door_status = "LOCKED"
                self._servo_angle = 0
                command = "LOCK"

            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "device_id": self._device_id,
                "claimed_identity": subject,
                "source_subject": source_subject,
                "recording": recording,
                "window_index": window_index,
                "authentication_decision": decision,
                "score_logit": float(score),
                "threshold": float(threshold),
                "command": command,
                "door_status": self._door_status,
                "servo_angle": self._servo_angle,
                "simulation": True,
            }

            self._last_command = command
            self._last_event = event
            self._history.insert(0, event)
            self._history = self._history[:100]

            return self._snapshot()


# One controller instance shared by the FastAPI application.
virtual_lock = VirtualIoTController()
