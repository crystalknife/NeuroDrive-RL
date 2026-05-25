"""
simulation/connection.py — CARLA server connection management.
Handles connect/disconnect with retry logic, sync mode, and cleanup.
"""

import sys
import time
import logging

try:
    import carla
except ImportError:
    print("[WARN] carla module not found. Install the CARLA Python API.")
    carla = None

import parameters as params

logger = logging.getLogger(__name__)


class CarlaConnection:
    """Manages connection to CARLA server."""

    def __init__(self, host=None, port=None, timeout=None):
        self.host = host or params.CARLA_HOST
        self.port = port or params.CARLA_PORT
        self.timeout = timeout or params.CARLA_TIMEOUT
        self.client = None
        self.world = None
        self.map = None
        self._original_settings = None

    def connect(self):
        """Connect to CARLA server with retries."""
        if carla is None:
            raise RuntimeError("CARLA Python API not installed.")

        for attempt in range(1, params.CONNECTION_RETRIES + 1):
            try:
                logger.info(f"Connecting to CARLA at {self.host}:{self.port} "
                            f"(attempt {attempt}/{params.CONNECTION_RETRIES})...")
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(self.timeout)
                self.world = self.client.get_world()
                self.map = self.world.get_map()

                # Store original settings
                self._original_settings = self.world.get_settings()

                # # Enable synchronous mode
                # if params.SYNC_MODE:
                #     settings = self.world.get_settings()
                #     settings.synchronous_mode = True
                #     settings.fixed_delta_seconds = params.FIXED_DELTA_SECONDS
                #     self.world.apply_settings(settings)

                # logger.info(f"Connected to CARLA server. Map: {self.map.name}")
                # return True

                settings = self.world.get_settings()

                settings.synchronous_mode = False   # 🔥 FORCE ASYNC
                settings.fixed_delta_seconds = None

                self.world.apply_settings(settings)
                logger.info(f"Connected to CARLA server. Map: {self.map.name}")
                return True

            except Exception as e:
                logger.warning(f"Connection attempt {attempt} failed: {e}")
                if attempt < params.CONNECTION_RETRIES:
                    time.sleep(2.0)
                else:
                    logger.error("All connection attempts failed.")
                    raise ConnectionError(
                        f"Cannot connect to CARLA at {self.host}:{self.port}"
                    ) from e

    def get_world(self):
        """Return the CARLA world object."""
        if self.world is None:
            raise RuntimeError("Not connected to CARLA. Call connect() first.")
        return self.world

    def get_map(self):
        """Return the CARLA map object."""
        if self.map is None:
            raise RuntimeError("Not connected to CARLA. Call connect() first.")
        return self.map

    def tick(self):
        """Advance the simulation by one step (sync mode)."""
        if self.world is not None and params.SYNC_MODE:
            self.world.tick()

    def cleanup(self):
        """Restore original settings and disconnect."""
        try:
            if self._original_settings is not None and self.world is not None:
                self.world.apply_settings(self._original_settings)
                logger.info("Restored original CARLA settings.")
        except Exception as e:
            logger.warning(f"Error restoring settings: {e}")
        self.client = None
        self.world = None
        self.map = None

    def __del__(self):
        self.cleanup()
