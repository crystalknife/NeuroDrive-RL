"""
simulation/sensors.py — Camera and collision sensor management for CARLA.
Thread-safe image buffering, collision detection, and robust cleanup.
"""

import queue
import logging
import numpy as np

try:
    import carla
except ImportError:
    carla = None

import parameters as params

logger = logging.getLogger(__name__)


class SensorManager:
    """Attach, manage, and clean up CARLA sensors on a vehicle."""

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = []

        # Image buffer (thread-safe, keeps only latest frame)
        self._image_queue = queue.Queue(maxsize=1)
        self._latest_image = None

        # Collision state
        self.collision_occurred = False
        self.collision_intensity = 0.0

        self._setup_camera()
        self._setup_collision()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _setup_camera(self):
        """Attach front-facing RGB camera."""
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(params.CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(params.CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(params.CAMERA_FOV))
        if params.SENSOR_TICK > 0:
            camera_bp.set_attribute("sensor_tick", str(params.SENSOR_TICK))

        # Mount position: slightly above and forward of vehicle center
        transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-15)
        )
        self.camera = self.world.spawn_actor(
            camera_bp, transform, attach_to=self.vehicle
        )
        self.camera.listen(self._camera_callback)
        self.sensors.append(self.camera)
        logger.debug("RGB camera attached.")

    def _camera_callback(self, image):
        """Store latest camera frame (thread-safe)."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((params.CAMERA_HEIGHT, params.CAMERA_WIDTH, 4))
        rgb = array[:, :, :3]  # drop alpha channel

        # Keep only the latest frame
        if not self._image_queue.empty():
            try:
                self._image_queue.get_nowait()
            except queue.Empty:
                pass
        self._image_queue.put(rgb)

    def get_image(self, timeout=2.0):
        """Retrieve the latest camera image. Returns (H, W, 3) uint8 array."""
        try:
            image = self._image_queue.get(timeout=timeout)
            self._latest_image = image
            return image
        except queue.Empty:
            logger.warning("Camera image timeout. Returning last known image.")
            if self._latest_image is not None:
                return self._latest_image
            # Return a black image as absolute fallback
            return np.zeros(
                (params.CAMERA_HEIGHT, params.CAMERA_WIDTH, 3), dtype=np.uint8
            )

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------
    def _setup_collision(self):
        """Attach collision sensor."""
        bp_lib = self.world.get_blueprint_library()
        collision_bp = bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle,
        )
        self.collision_sensor.listen(self._collision_callback)
        self.sensors.append(self.collision_sensor)
        logger.debug("Collision sensor attached.")

    def _collision_callback(self, event):
        """Record collision event."""
        impulse = event.normal_impulse
        intensity = (impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5
        self.collision_occurred = True
        self.collision_intensity = max(self.collision_intensity, intensity)
        logger.debug(f"Collision detected — intensity: {intensity:.1f}")

    def reset_collision(self):
        """Clear collision state for new episode."""
        self.collision_occurred = False
        self.collision_intensity = 0.0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def destroy(self):
        """Destroy all sensors. Per-sensor error handling."""
        for sensor in self.sensors:
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception as e:
                logger.warning(f"Error destroying sensor {sensor}: {e}")
        self.sensors.clear()
        logger.debug("All sensors destroyed.")
