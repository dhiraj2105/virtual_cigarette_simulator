from __future__ import annotations

import math
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

WINDOW_NAME = "Virtual Cigarette Simulator By Dhiraj"
CAMERA_INDEX = 0
CAMERA_WIDTH = 1500
CAMERA_HEIGHT = 900
CAMERA_FPS = 60

PINCH_MAX_RATIO = 0.62
FINGER_EXTENSION_RATIO = 1.15
MOUTH_DISTANCE_RATIO = 0.22
CIGARETTE_LENGTH = 105
FILTER_LENGTH = 25
CIGARETTE_RADIUS = 7
SMOKE_MAX_PARTICLES = 100
SMOKE_SPAWN_INTERVAL = 0.045

# Inhale/exhale interaction.
INHALE_MIN_DURATION = 0.55
INHALE_MAX_DURATION = 3.0
EXHALE_COOLDOWN = 0.25
MOUTH_OPEN_RATIO = 0.035
MOUTH_OPEN_HYSTERESIS = 0.008
EXHALE_PARTICLES_PER_SECOND = 200
EXHALE_DURATION = 5

WHITE = (245, 245, 245)
LIGHT_GRAY = (205, 205, 205)
GRAY = (125, 125, 125)
DARK_GRAY = (55, 55, 55)
BLACK = (15, 15, 15)
RED = (45, 55, 235)
ORANGE = (40, 150, 255)
YELLOW = (80, 220, 255)


Point = tuple[int, int]


@dataclass
class HandState:
    holder: Point
    direction: tuple[float, float]
    confidence: float


@dataclass
class SmokeParticle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    life: float
    max_life: float
    phase: float

    @property
    def alpha(self) -> float:
        return max(0.0, min(1.0, self.life / self.max_life))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit_vector(a: Point, b: Point) -> tuple[float, float]:
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, -1.0
    return dx / length, dy / length


def landmark_point(landmark, width: int, height: int) -> Point:
    return (
        int(clamp(float(landmark.x), 0.0, 1.0) * (width - 1)),
        int(clamp(float(landmark.y), 0.0, 1.0) * (height - 1)),
    )


def alpha_circle(image, center: Point, radius: int, color, alpha: float) -> None:
    if radius <= 0 or alpha <= 0.0:
        return
    overlay = image.copy()
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, clamp(alpha, 0.0, 1.0), image, 1.0 - clamp(alpha, 0.0, 1.0), 0, image)


def download_model(path: Path, url: str, description: str) -> None:
    if path.exists() and path.stat().st_size > 100_000:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".download")

    print(f"Downloading {description} model...")
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Virtual-Cigarette-Simulator/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, temp_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

        if temp_path.stat().st_size <= 100_000:
            raise RuntimeError(f"Downloaded {description} model is unexpectedly small.")

        temp_path.replace(path)
        print(f"Downloaded: {path.name}")
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the {description} model.\n"
            f"Please check your internet connection and run the application again.\n"
            f"Model URL: {url}\n"
            f"Original error: {exc}"
        ) from exc


def ensure_models() -> None:
    download_model(HAND_MODEL, HAND_MODEL_URL, "hand landmarker")
    download_model(FACE_MODEL, FACE_MODEL_URL, "face landmarker")


class VisionTracker:
    """MediaPipe Tasks API tracker for hands and face landmarks."""

    def __init__(self) -> None:
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.55,
            min_face_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        self.timestamp_ms = 0

    def process(self, frame):
        """Process one BGR OpenCV frame using the Tasks VIDEO API."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.timestamp_ms += 1
        hands = self.hand_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        face = self.face_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        return hands, face

    def close(self) -> None:
        self.hand_landmarker.close()
        self.face_landmarker.close()


def detect_cigarette_hand(hand, width: int, height: int) -> Optional[HandState]:
    lm = hand

    wrist = landmark_point(lm[0], width, height)
    index_mcp = landmark_point(lm[5], width, height)
    index_pip = landmark_point(lm[6], width, height)
    index_tip = landmark_point(lm[8], width, height)
    middle_mcp = landmark_point(lm[9], width, height)
    middle_pip = landmark_point(lm[10], width, height)
    middle_tip = landmark_point(lm[12], width, height)

    palm_size = max(distance(wrist, middle_mcp), distance(index_mcp, middle_mcp), 1.0)
    fingertip_gap_ratio = distance(index_tip, middle_tip) / palm_size

    index_extension = distance(wrist, index_tip) / max(distance(wrist, index_mcp), 1.0)
    middle_extension = distance(wrist, middle_tip) / max(distance(wrist, middle_mcp), 1.0)

    valid = (
        fingertip_gap_ratio <= PINCH_MAX_RATIO
        and index_extension >= FINGER_EXTENSION_RATIO
        and middle_extension >= FINGER_EXTENSION_RATIO
        and distance(wrist, index_tip) > distance(wrist, index_pip)
        and distance(wrist, middle_tip) > distance(wrist, middle_pip)
    )

    if not valid:
        return None

    holder = (
        (index_tip[0] + middle_tip[0]) // 2,
        (index_tip[1] + middle_tip[1]) // 2,
    )
    palm_center = (
        (wrist[0] + index_mcp[0] + middle_mcp[0]) // 3,
        (wrist[1] + index_mcp[1] + middle_mcp[1]) // 3,
    )

    return HandState(
        holder=holder,
        direction=unit_vector(palm_center, holder),
        confidence=clamp(1.0 - fingertip_gap_ratio / PINCH_MAX_RATIO, 0.0, 1.0),
    )


def get_mouth_state(face_landmarks, width: int, height: int) -> tuple[Point, float, float, float]:
    """Return mouth center, face scale, mouth openness and mouth width."""
    upper = landmark_point(face_landmarks[13], width, height)
    lower = landmark_point(face_landmarks[14], width, height)
    left = landmark_point(face_landmarks[61], width, height)
    right = landmark_point(face_landmarks[291], width, height)
    top = landmark_point(face_landmarks[10], width, height)
    bottom = landmark_point(face_landmarks[152], width, height)

    mouth = (
        (upper[0] + lower[0] + left[0] + right[0]) // 4,
        (upper[1] + lower[1] + left[1] + right[1]) // 4,
    )
    face_scale = max(distance(top, bottom), 1.0)
    mouth_width = max(distance(left, right), 1.0)
    mouth_open_ratio = distance(upper, lower) / face_scale

    return mouth, face_scale, mouth_open_ratio, mouth_width


def draw_cigarette(image, holder: Point, direction: tuple[float, float], smoking: bool) -> Point:
    dx, dy = direction
    tip = (int(holder[0] + dx * CIGARETTE_LENGTH), int(holder[1] + dy * CIGARETTE_LENGTH))
    body_end = (int(tip[0] - dx * FILTER_LENGTH), int(tip[1] - dy * FILTER_LENGTH))

    cv2.line(image, holder, body_end, DARK_GRAY, CIGARETTE_RADIUS * 2 + 2, cv2.LINE_AA)
    cv2.line(image, holder, body_end, WHITE, CIGARETTE_RADIUS * 2, cv2.LINE_AA)

    filter_end = (int(body_end[0] + dx * FILTER_LENGTH), int(body_end[1] + dy * FILTER_LENGTH))
    cv2.line(image, body_end, filter_end, (85, 145, 195), CIGARETTE_RADIUS * 2, cv2.LINE_AA)

    for fraction in (0.25, 0.55, 0.82):
        px = int(body_end[0] + (filter_end[0] - body_end[0]) * fraction)
        py = int(body_end[1] + (filter_end[1] - body_end[1]) * fraction)
        cv2.circle(image, (px, py), 1, (125, 180, 220), -1, cv2.LINE_AA)

    alpha_circle(image, tip, 10, ORANGE, 0.24)
    cv2.circle(image, tip, 5, RED, -1, cv2.LINE_AA)
    cv2.circle(image, tip, 3, YELLOW, -1, cv2.LINE_AA)

    if smoking:
        alpha_circle(image, tip, 15, ORANGE, 0.18)

    return tip


class SmokeSystem:
    """CPU-only smoke particles for cigarette and mouth exhalation."""

    def __init__(self) -> None:
        self.particles: list[SmokeParticle] = []
        self.accumulator = 0.0
        self.exhale_accumulator = 0.0

    def reset(self) -> None:
        self.particles.clear()
        self.accumulator = 0.0
        self.exhale_accumulator = 0.0

    def spawn(self, origin: Point, intensity: float, direction: tuple[float, float] = (0.0, -1.0)) -> None:
        if len(self.particles) >= SMOKE_MAX_PARTICLES:
            return

        dx, dy = direction
        spread = 5.0 + 4.0 * intensity
        self.particles.append(
            SmokeParticle(
                x=origin[0] + random.uniform(-spread, spread),
                y=origin[1] + random.uniform(-spread, spread),
                vx=dx * random.uniform(18.0, 36.0) + random.uniform(-8.0, 8.0),
                vy=dy * random.uniform(18.0, 36.0) + random.uniform(-8.0, 8.0),
                radius=random.uniform(3.5, 7.0) * (0.8 + intensity * 0.25),
                life=random.uniform(0.8, 1.5),
                max_life=random.uniform(0.8, 1.5),
                phase=random.uniform(0, math.tau),
            )
        )

    def update_cigarette_smoke(
        self,
        dt: float,
        origin: Optional[Point],
        active: bool,
        intensity: float,
    ) -> None:
        """Create smoke from the cigarette only when it is actively burning."""
        if active and origin is not None:
            self.accumulator += dt
            while self.accumulator >= SMOKE_SPAWN_INTERVAL:
                self.accumulator -= SMOKE_SPAWN_INTERVAL
                self.spawn(origin, intensity, (0.0, -1.0))

    def update_exhale(
        self,
        dt: float,
        mouth: Optional[Point],
        active: bool,
        direction: tuple[float, float],
        intensity: float = 1.0,
    ) -> None:
        """Emit a burst of smoke from the mouth during an exhale."""
        if not active or mouth is None:
            self.exhale_accumulator = 0.0
            return

        self.exhale_accumulator += dt * EXHALE_PARTICLES_PER_SECOND * intensity
        while self.exhale_accumulator >= 1.0:
            self.exhale_accumulator -= 1.0
            if len(self.particles) >= SMOKE_MAX_PARTICLES:
                break
            self.spawn(mouth, 1.35 * intensity, direction)

    def update_particles(self, dt: float) -> None:
        alive: list[SmokeParticle] = []

        for particle in self.particles:
            particle.life -= dt
            if particle.life <= 0:
                continue

            particle.phase += dt * 2.2
            particle.vx += math.sin(particle.phase) * 4.0 * dt
            particle.vy -= 5.0 * dt
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.radius += 5.0 * dt
            alive.append(particle)

        self.particles = alive

    def draw(self, image) -> None:
        for particle in self.particles:
            alpha_circle(
                image,
                (int(particle.x), int(particle.y)),
                max(1, int(particle.radius)),
                LIGHT_GRAY,
                particle.alpha * 0.38,
            )


def draw_hud(
    image,
    fps: float,
    holding: bool,
    smoking: bool,
    inhaling: bool,
    ready_to_exhale: bool,
    exhaling: bool,
    show_fps: bool,
) -> None:
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (14, 14), (w - 14, 126), BLACK, -1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)

    cv2.putText(image, "DON'T SMOKE", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, WHITE, 2, cv2.LINE_AA)

    if exhaling:
        status, status_color = "EXHALING SMOKE", YELLOW
    elif inhaling:
        status, status_color = "INHALING...", YELLOW
    elif ready_to_exhale:
        status, status_color = "INHALED - OPEN MOUTH TO EXHALE", WHITE
    elif smoking:
        status, status_color = "CIGARETTE AT MOUTH", WHITE
    elif holding:
        status, status_color = "CIGARETTE HELD", WHITE
    else:
        status, status_color = "Bring index + middle fingers together", LIGHT_GRAY

    cv2.putText(image, status, (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)
    # cv2.putText(image, "Hold at mouth + open to inhale | Move away | Open to exhale", (30, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY, 1, cv2.LINE_AA)
    cv2.putText(image, "Q / ESC Quit    R Reset    F FPS", (30, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY, 1, cv2.LINE_AA)

    if show_fps:
        cv2.putText(image, f"{fps:.1f} FPS", (w - 105, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, LIGHT_GRAY, 1, cv2.LINE_AA)


def open_camera() -> cv2.VideoCapture:
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and ensure another application is not using it.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    return camera


def main() -> None:
    print("Virtual Cigarette Simulator")
    print()

    ensure_models()
    tracker = VisionTracker()
    camera = open_camera()
    smoke = SmokeSystem()

    show_fps = True
    previous_time = time.perf_counter()
    fps = 0.0

    # Interaction state.
    inhale_started_at: Optional[float] = None
    inhale_completed = False
    inhale_strength = 0.0
    exhale_started_at: Optional[float] = None
    last_exhale_time = -999.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam.")

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            hand_result, face_result = tracker.process(frame)

            cigarette: Optional[HandState] = None
            for hand in hand_result.hand_landmarks:
                candidate = detect_cigarette_hand(hand, width, height)
                if candidate is not None and (cigarette is None or candidate.confidence > cigarette.confidence):
                    cigarette = candidate

            mouth_center: Optional[Point] = None
            face_scale = 1.0
            mouth_open_ratio = 0.0
            mouth_width = 1.0

            if face_result.face_landmarks:
                (
                    mouth_center,
                    face_scale,
                    mouth_open_ratio,
                    mouth_width,
                ) = get_mouth_state(face_result.face_landmarks[0], width, height)

            now = time.perf_counter()
            dt = clamp(now - previous_time, 0.001, 0.08)
            previous_time = now
            instant_fps = 1.0 / dt
            fps = instant_fps if fps == 0 else fps * 0.90 + instant_fps * 0.10

            holding = cigarette is not None
            cigarette_at_mouth = False
            cigarette_tip: Optional[Point] = None

            if cigarette is not None:
                dx, dy = cigarette.direction
                preview_tip = (
                    int(cigarette.holder[0] + dx * CIGARETTE_LENGTH),
                    int(cigarette.holder[1] + dy * CIGARETTE_LENGTH),
                )

                if mouth_center is not None:
                    mouth_distance = distance(preview_tip, mouth_center)
                    cigarette_at_mouth = mouth_distance <= face_scale * MOUTH_DISTANCE_RATIO

                cigarette_tip = draw_cigarette(
                    frame,
                    cigarette.holder,
                    cigarette.direction,
                    cigarette_at_mouth,
                )

            # Use hysteresis so tiny landmark jitter does not repeatedly toggle
            # the mouth state around the threshold.
            mouth_is_open = mouth_open_ratio >= MOUTH_OPEN_RATIO
            mouth_is_closed = mouth_open_ratio <= MOUTH_OPEN_RATIO - MOUTH_OPEN_HYSTERESIS

            # ---------------------------------------------------------------
            # INHALE
            # ---------------------------------------------------------------
            # The user must actually have the cigarette at the mouth and keep
            # the mouth open for a short continuous period.
            if cigarette_at_mouth and mouth_is_open and not inhale_completed:
                if inhale_started_at is None:
                    inhale_started_at = now

                inhale_duration = now - inhale_started_at
                inhale_strength = clamp(
                    (inhale_duration - INHALE_MIN_DURATION)
                    / max(INHALE_MAX_DURATION - INHALE_MIN_DURATION, 0.001),
                    0.0,
                    1.0,
                )

                if inhale_duration >= INHALE_MIN_DURATION:
                    inhale_completed = True
                    inhale_strength = max(inhale_strength, 0.35)
            elif not cigarette_at_mouth:
                inhale_started_at = None

            # If the cigarette leaves the mouth before a completed inhale,
            # cancel the incomplete inhale.
            if not cigarette_at_mouth and not inhale_completed:
                inhale_started_at = None
                inhale_strength = 0.0

            # ---------------------------------------------------------------
            # EXHALE
            # ---------------------------------------------------------------
            # After a successful inhale, the cigarette must be away from the
            # mouth. Opening the mouth then creates smoke travelling outward.
            can_exhale = (
                inhale_completed
                and not cigarette_at_mouth
                and mouth_is_open
                and now - last_exhale_time >= EXHALE_COOLDOWN
            )

            exhaling = False
            if can_exhale:
                if exhale_started_at is None:
                    exhale_started_at = now
                    last_exhale_time = now

                exhale_duration = now - exhale_started_at
                exhaling = exhale_duration <= EXHALE_DURATION

                # Exhale direction follows the vector from the face center
                # toward the mouth, with a slight upward component.
                if mouth_center is not None:
                    face_center = (
                        int(width * 0.5),
                        int(mouth_center[1] - face_scale * 0.20),
                    )
                    exhale_direction = unit_vector(face_center, mouth_center)
                else:
                    exhale_direction = (0.0, 1.0)

                smoke.update_exhale(
                    dt,
                    mouth_center,
                    exhaling,
                    exhale_direction,
                    intensity=0.8 + inhale_strength * 0.8,
                )

                if not exhaling:
                    inhale_completed = False
                    inhale_started_at = None
                    exhale_started_at = None
                    inhale_strength = 0.0
            else:
                exhale_started_at = None

            # If the mouth closes after inhaling, keep the inhale stored.
            # This makes the interaction feel natural: inhale, close mouth,
            # move cigarette away, then open to exhale.
            ready_to_exhale = inhale_completed and not exhaling

            # Cigarette smoke rises normally when the cigarette is away from
            # the mouth. During an inhale, smoke travels from the cigarette
            # tip toward the mouth so the user can visibly see the inhale.
            cigarette_smoke_active = holding and not cigarette_at_mouth and not inhale_completed
            smoke.update_cigarette_smoke(
                dt,
                cigarette_tip,
                cigarette_smoke_active,
                0.65,
            )

            if (
                cigarette_at_mouth
                and inhale_started_at is not None
                and not inhale_completed
                and cigarette_tip is not None
                and mouth_center is not None
            ):
                inhale_direction = unit_vector(cigarette_tip, mouth_center)
                smoke.update_cigarette_smoke(
                    dt,
                    cigarette_tip,
                    True,
                    0.45,
                )
                # Re-orient the newest particles toward the mouth. This is
                # intentionally done after spawning so only inhale particles
                # are pulled toward the mouth.
                for particle in smoke.particles[-3:]:
                    particle.vx = inhale_direction[0] * 35.0 + random.uniform(-5.0, 5.0)
                    particle.vy = inhale_direction[1] * 35.0 + random.uniform(-5.0, 5.0)

            smoke.update_particles(dt)
            smoke.draw(frame)

            # Inhale visual feedback.
            if cigarette_at_mouth and inhale_started_at is not None and mouth_center is not None:
                progress = clamp(
                    (now - inhale_started_at) / INHALE_MIN_DURATION,
                    0.0,
                    1.0,
                )
                cv2.circle(
                    frame,
                    mouth_center,
                    max(8, int(face_scale * 0.028)),
                    YELLOW,
                    2,
                    cv2.LINE_AA,
                )
                bar_width = int(face_scale * 0.30)
                bar_x = mouth_center[0] - bar_width // 2
                bar_y = mouth_center[1] + int(face_scale * 0.055)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 5), DARK_GRAY, -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 5), YELLOW, -1)

            # Exhale visual cue around the mouth.
            if exhaling and mouth_center is not None:
                alpha_circle(
                    frame,
                    mouth_center,
                    max(10, int(face_scale * 0.045)),
                    LIGHT_GRAY,
                    0.12,
                )

            draw_hud(
                frame,
                fps,
                holding,
                cigarette_at_mouth,
                inhale_started_at is not None and not inhale_completed,
                ready_to_exhale,
                exhaling,
                show_fps,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                smoke.reset()
                inhale_started_at = None
                inhale_completed = False
                inhale_strength = 0.0
                exhale_started_at = None
            if key == ord("f"):
                show_fps = not show_fps

    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
