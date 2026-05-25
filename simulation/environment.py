"""
simulation/environment.py — Gym-like CARLA driving environment.

Features:
- Waypoint-based navigation with GlobalRoutePlanner
- Lookahead direction for curve anticipation
- Alignment-based reward (dot product) with cross-product steering
- Symmetric 7-term reward (no directional bias)
- Action smoothing (exponential filter)
- Curriculum learning (5 levels)
- Fault-tolerant reset/step
"""
import sys

sys.path.append(r"D:\car\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")
sys.path.append(r"D:\car\WindowsNoEditor\PythonAPI\carla")

import math
import random
import logging
import numpy as np

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import parameters as params
from simulation.sensors import SensorManager

logger = logging.getLogger(__name__)


class CarlaEnvironment:
    """Custom Gym-like environment for autonomous driving in CARLA."""

    def __init__(self, connection, difficulty=1):
        self.conn = connection
        self.world = connection.get_world()
        self.map = connection.get_map()
        self.difficulty = max(1, min(5, difficulty))

        # Route planner
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

        print("DEBUG GlobalRoutePlanner:", GlobalRoutePlanner)

        dao = GlobalRoutePlannerDAO(self.map, params.ROUTE_RESOLUTION)
        self.planner = GlobalRoutePlanner(dao)
        self.planner.setup()

        # State
        self.vehicle = None
        self.sensor_manager = None
        self.route = []
        self.current_wp_idx = 0
        self.prev_steer = 0.0
        self.prev_dist_center = 0.0
        self.prev_throttle = 0.0
        self.episode_step = 0

        # Blueprint
        self.bp_lib = self.world.get_blueprint_library()

    # ==================================================================
    # Reset
    # ==================================================================
    def reset(self):
        """
        Reset the environment for a new episode.
        Returns observation dict or None on failure (caller should skip).
        """
        try:
            self._cleanup()
            self._spawn_vehicle()
            self.sensor_manager = SensorManager(self.world, self.vehicle)

            # Generate route based on curriculum difficulty
            self._generate_route()

            self.prev_steer = 0.0
            self.prev_dist_center = 0.0
            self.episode_step = 0

            # Let simulation settle
            for _ in range(5):
                self.conn.tick()

            return self._get_observation()

        except Exception as e:
            logger.error(f"[RESET FAILED] {e}")
            print(f"[WARN] Reset failed: {e}")
            self._cleanup()
            return None

    # ==================================================================
    # Step
    # ==================================================================
    def step(self, action):
        """
        Execute action [steer, throttle] in the environment.
        Returns (observation, reward, done, info).
        On failure: returns (last_valid_obs, 0, True, {"error": ...}).
        """
        try:
            self.episode_step += 1

            # Decode action
            raw_steer = float(np.clip(action[0], -1.0, 1.0))
            raw_throttle = float(np.clip(action[1], 0.0, 1.0))

            # 🚨 CONTROL FIX: reduce speed dominance
            raw_throttle = float(np.clip(action[1], 0.0, 1.0))

            # Smooth throttle
            alpha_t = 0.8
            smoothed_throttle = alpha_t * self.prev_throttle + (1 - alpha_t) * raw_throttle

            self.prev_throttle = smoothed_throttle

            # Final throttle
            throttle = max(0.3, smoothed_throttle * 0.5)

            # Action smoothing (exponential filter on steering)
            smoothed_steer = (
                (1.0 - params.STEER_SMOOTHING_ALPHA) * self.prev_steer
                + params.STEER_SMOOTHING_ALPHA * raw_steer
            )

            # Remove tiny jitter
            if abs(smoothed_steer) < 0.05:
                smoothed_steer = 0.0

            control = carla.VehicleControl(
                steer=smoothed_steer,
                throttle=throttle,
                brake=0.0,
            )
           # Apply control FIRST
            self.vehicle.apply_control(control)

            # Sync with simulator frame
            self.world.wait_for_tick()

            # Update camera AFTER frame update (IMPORTANT)
            if self.camera is not None:
                self.world.get_spectator().set_transform(
                    self.camera.get_transform()
                )

            # Small delay for smooth rendering
            import time
            time.sleep(0.03)
            # time.sleep(0.05)   # 🔥 THIS IS CRITICAL

            # try:
            #     spectator = self.world.get_spectator()
            #     transform = self.vehicle.get_transform()

            #     forward = transform.get_forward_vector()

            #     spectator.set_transform(carla.Transform(
            #         transform.location - forward * 12 + carla.Location(z=6),
            #         carla.Rotation(pitch=-25, yaw=transform.rotation.yaw)
            #     ))

            # except Exception as e:
            #     print("Camera update failed:", e)
            # Compute reward components
            reward, info = self._compute_reward(smoothed_steer)

            # Update state tracking for next step
            self.prev_steer = smoothed_steer
            self.prev_dist_center = info.get("dist_center", 0.0)

            # Advance waypoint
            self._advance_waypoint()

            # Check termination
            done = self._check_done(info)

            obs = self._get_observation()
            return obs, reward, done, info

        except Exception as e:
            logger.error(f"[STEP FAILED] t={self.episode_step}: {e}")
            print(f"[WARN] Step failed at t={self.episode_step}: {e}")
            zero_obs = self._zero_observation()
            return zero_obs, 0.0, True, {"error": str(e)}
        print(self.vehicle.get_location())

    # ==================================================================
    # Spawning
    # ==================================================================
    def _spawn_vehicle(self):
        """Spawn ego vehicle at a random spawn point."""
        vehicle_bp = self.bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = self.map.get_spawn_points()

        if not spawn_points:
            raise RuntimeError("No spawn points available on map.")

        spawn_tf = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_tf)
        # ===== ATTACH DISPLAY CAMERA (FOR CARLA WINDOW) =====
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')

        # Attach slightly behind and above the car
        camera_transform = carla.Transform(
            carla.Location(x=-6, z=3),
            carla.Rotation(pitch=-15)
        )

        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )

        # ===== STEP 2: FORCE SPECTATOR TO FOLLOW CAMERA =====
        self.world.get_spectator().set_transform(
            self.camera.get_transform()
        )

        logger.info(f"Vehicle spawned at {spawn_tf.location}")
        # spectator = self.world.get_spectator()
        # transform = self.vehicle.get_transform()

        # spectator = self.world.get_spectator()

        # for _ in range(10):
        #     spectator.set_transform(
        #         carla.Transform(
        #             self.vehicle.get_location() + carla.Location(z=10),
        #             carla.Rotation(pitch=-45)
        #         )
        #     )
        #     self.conn.tick()

    # ==================================================================
    # Route Generation (Curriculum)
    # ==================================================================
    def _generate_route(self):
        """Generate waypoint route based on difficulty level."""
        level_cfg = params.CURRICULUM_LEVELS.get(
            self.difficulty, params.CURRICULUM_LEVELS[3]
        )
        min_wp = level_cfg["min_wp"]
        max_wp = level_cfg["max_wp"]
        target_len = random.randint(min_wp, max_wp)

        spawn_loc = self.vehicle.get_location()
        spawn_points = self.map.get_spawn_points()

        # Try to find a destination that gives the desired route length
        best_route = []
        for _ in range(10):  # 10 attempts
            dest_tf = random.choice(spawn_points)
            try:
                route = self.planner.trace_route(spawn_loc, dest_tf.location)
                if len(route) >= min_wp:
                    if len(route) <= max_wp or len(best_route) == 0:
                        best_route = route
                    if min_wp <= len(route) <= max_wp:
                        break
            except Exception as e:
                logger.debug(f"Route planning attempt failed: {e}")
                continue

        if not best_route:
            # Fallback: use waypoints from current position
            current_wp = self.map.get_waypoint(spawn_loc)
            best_route = [(current_wp, None)]
            for _ in range(target_len):
                nexts = best_route[-1][0].next(params.ROUTE_RESOLUTION)
                if nexts:
                    best_route.append((nexts[0], None))
                else:
                    break

        self.route = best_route
        self.current_wp_idx = 0
        logger.info(f"Route generated: {len(self.route)} waypoints "
                    f"(difficulty {self.difficulty})")

    # ==================================================================
    # Observation
    # ==================================================================
    def _get_observation(self):
        """
        Build observation dict:
        - image: (H, W, 3) uint8
        - navigation: [speed, dist_center, angle_road, dist_next_wp, dx, dy, curvature]
        """
        image = self.sensor_manager.get_image()

        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

        

        dist_center = self._distance_from_center()
        angle_road = self._angle_to_road()
        dist_next_wp, dx, dy = self._waypoint_direction()
        curvature = self._estimate_curvature()

        navigation = np.array([
            speed / params.MAX_SPEED,       # normalize speed
            dist_center,                     # meters (can be negative)
            angle_road / math.pi,           # normalize to [-1, 1]
            min(dist_next_wp / 50.0, 1.0),  # normalize distance
            dx, dy,                          # normalized direction
            curvature,                       # already normalized by π
        ], dtype=np.float32)

        return {"image": image, "navigation": navigation}

    def _zero_observation(self):
        """Return a zero observation (used only for error-return, never for PPO)."""
        return {
            "image": np.zeros(
                (params.CAMERA_HEIGHT, params.CAMERA_WIDTH, 3), dtype=np.uint8
            ),
            "navigation": np.zeros(params.NAV_FEATURE_DIM, dtype=np.float32),
        }

    # ==================================================================
    # Navigation Calculations
    # ==================================================================
    def _distance_from_center(self):
        """Lateral distance from the lane center."""
        vehicle_loc = self.vehicle.get_location()
        nearest_wp = self.map.get_waypoint(
            vehicle_loc, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if nearest_wp is None:
            return 0.0

        # Project vehicle position onto waypoint frame
        wp_loc = nearest_wp.transform.location
        wp_fwd = nearest_wp.transform.get_forward_vector()

        # Right vector (perpendicular to forward in XY plane)
        right_x = -wp_fwd.y
        right_y = wp_fwd.x

        dx = vehicle_loc.x - wp_loc.x
        dy = vehicle_loc.y - wp_loc.y

        lateral_dist = dx * right_x + dy * right_y
        return lateral_dist

    def _angle_to_road(self):
        """Signed angle between vehicle heading and road direction (radians)."""
        vehicle_tf = self.vehicle.get_transform()
        vehicle_fwd = vehicle_tf.get_forward_vector()

        nearest_wp = self.map.get_waypoint(
            vehicle_tf.location, project_to_road=True
        )
        if nearest_wp is None:
            return 0.0

        wp_fwd = nearest_wp.transform.get_forward_vector()

        # Angle via atan2
        cross = vehicle_fwd.x * wp_fwd.y - vehicle_fwd.y * wp_fwd.x
        dot = vehicle_fwd.x * wp_fwd.x + vehicle_fwd.y * wp_fwd.y
        angle = math.atan2(cross, dot)
        return angle

    def _waypoint_direction(self):
        """
        Lookahead direction: normalized vector from vehicle to lookahead waypoint.
        Returns (distance, dx, dy).
        """
        if not self.route:
            return 0.0, 0.0, 1.0

        # Robust lookahead index
        lookahead_idx = min(
            self.current_wp_idx + params.LOOKAHEAD_STEPS,
            len(self.route) - 1
        )
        # Ensure lookahead is always ahead of current
        if lookahead_idx <= self.current_wp_idx:
            lookahead_idx = min(self.current_wp_idx + 1, len(self.route) - 1)

        target_wp = self.route[lookahead_idx][0]
        vehicle_loc = self.vehicle.get_location()
        wp_loc = target_wp.transform.location

        dx = wp_loc.x - vehicle_loc.x
        dy = wp_loc.y - vehicle_loc.y
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 1e-6:
            dx /= dist
            dy /= dist
        else:
            dx, dy = 0.0, 1.0

        # remove tiny noise
        if abs(dx) < 0.02:
            dx = 0.0
        if abs(dy) < 0.02:
            dy = 0.0

        return dist, dx, dy

    def _estimate_curvature(self):
        """
        Curvature from angle between 3 consecutive waypoints.
        Normalized by π → [0, 1].
        """
        if len(self.route) < 3:
            return 0.0

        idx = min(self.current_wp_idx, len(self.route) - 3)

        p0 = self.route[idx][0].transform.location
        p1 = self.route[idx + 1][0].transform.location
        p2 = self.route[idx + 2][0].transform.location

        v1x = p1.x - p0.x
        v1y = p1.y - p0.y
        v2x = p2.x - p1.x
        v2y = p2.y - p1.y

        len1 = math.sqrt(v1x**2 + v1y**2) + 1e-8
        len2 = math.sqrt(v2x**2 + v2y**2) + 1e-8

        cos_angle = (v1x * v2x + v1y * v2y) / (len1 * len2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)

        curvature = angle / math.pi  # normalize to [0, 1]
        return curvature

    # ==================================================================
    # Reward
    # ==================================================================
    def _compute_reward(self, applied_steer):
        info = {}
        reward = 0.0

        vehicle_fwd = self.vehicle.get_transform().get_forward_vector()
        _, la_dx, la_dy = self._waypoint_direction()

        # 1. Alignment (MAIN SIGNAL)
        alignment = vehicle_fwd.x * la_dx + vehicle_fwd.y * la_dy
        reward += 2.0 * alignment
        info["alignment"] = alignment

        # 2. Lane centering
        dist_center = self._distance_from_center()
        clamped = max(-4.0, min(4.0, dist_center))
        reward -= 0.3 * (clamped ** 2)
        info["dist_center"] = dist_center

        # 3. Collision
        if self.sensor_manager.collision_occurred:
            reward += params.REWARD_COLLISION

        # 4. Alive bonus
        reward += 0.2

        # 5. Speed reward
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        reward += 0.05 * speed

        reward *= params.REWARD_SCALE
        info["raw_reward"] = reward

        return reward, info

    # ==================================================================
    # Waypoint Advancement
    # ==================================================================
    def _advance_waypoint(self):
        """Advance current waypoint index if vehicle is close enough."""
        if not self.route or self.current_wp_idx >= len(self.route):
            return

        vehicle_loc = self.vehicle.get_location()
        wp_loc = self.route[self.current_wp_idx][0].transform.location
        dist = math.sqrt(
            (vehicle_loc.x - wp_loc.x)**2 + (vehicle_loc.y - wp_loc.y)**2
        )

        if dist < params.WAYPOINT_THRESHOLD:
            self.current_wp_idx += 1

    # ==================================================================
    # Termination
    # ==================================================================
    def _check_done(self, info):
        """Check if episode should terminate."""
        # Collision
        if self.sensor_manager.collision_occurred:
            logger.info("Episode ended: collision.")
            return True

        # Route complete
        if self.current_wp_idx >= len(self.route):
            logger.info("Episode ended: route complete.")
            return True

        # Max timesteps
        if self.episode_step >= params.MAX_TIMESTEPS_PER_EPISODE:
            logger.info("Episode ended: max timesteps.")
            return True

        # Off-road check (too far from center)
        if abs(self._distance_from_center()) > 4.0:
            logger.info("Episode ended: off road.")
            return True

        return False

    # ==================================================================
    # Cleanup
    # ==================================================================
    def _cleanup(self):
        """Destroy sensors and vehicle. Per-actor error handling."""
        if self.sensor_manager is not None:
            try:
                self.sensor_manager.destroy()
            except Exception as e:
                logger.warning(f"Error destroying sensors: {e}")
            self.sensor_manager = None

        if self.vehicle is not None:
            try:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
            except Exception as e:
                logger.warning(f"Error destroying vehicle: {e}")
            self.vehicle = None

        self.route = []
        self.current_wp_idx = 0
        self.prev_dist_center = 0.0

    def set_difficulty(self, level):
        """Update curriculum difficulty level."""
        self.difficulty = max(1, min(5, level))
        logger.info(f"Difficulty set to {self.difficulty}: "
                    f"{params.CURRICULUM_LEVELS[self.difficulty]['description']}")

    def close(self):
        """Clean shutdown."""
        self._cleanup()
