"""HeadCC.py — track head pose and map to MIDI CC messages.

This module started life as a single monolithic script.  The behaviour stays the
same, but the code is now structured in cohesive classes that encapsulate
configuration handling, pose estimation, MIDI I/O and UI management.  Each unit
has a clear set of responsibilities which keeps the main application loop small
and easier to reason about.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
from mediapipe import solutions as mp_solutions
from mediapipe.framework.formats import landmark_pb2
from mido import Message, get_output_names, open_output

CONFIG_PATH = "headcc_cfg.json"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ema(prev: Optional[float], new: float, alpha: float) -> float:
    return new if prev is None else alpha * new + (1 - alpha) * prev


@dataclass
class AppDefaults:
    midi_port_substr: str = "HeadCC"
    midi_channel: int = 0  # 0..15
    cc_yaw: int = 1
    cc_pitch: int = 11
    cc_roll: int = 74
    cc_yaw_abs: int = 21
    cc_pitch_abs: int = 22
    cc_roll_abs: int = 23
    send_rate_hz: int = 30
    smooth_alpha: float = 0.25
    deadzone_deg: float = 2.0
    yaw_range_deg: Tuple[float, float] = (-45.0, 45.0)
    pitch_range_deg: Tuple[float, float] = (-30.0, 30.0)
    roll_range_deg: Tuple[float, float] = (-35.0, 35.0)
    global_sens_pct: int = 100
    invert_yaw: bool = False
    invert_pitch: bool = False
    invert_roll: bool = False
    cam_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    fov_deg: float = 60.0
    ui_scale: float = 1.0


DEFAULTS = AppDefaults()
UI_SCALE_MIN = 0.6
UI_SCALE_MAX = 2.5

LM = [1, 152, 33, 263, 61, 291]  # nose, chin, eye outer L/R, mouth L/R
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float32,
)


class ConfigManager:
    """Wrapper around the JSON configuration file."""

    def __init__(self, path: str = CONFIG_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        defaults = DEFAULTS.__dict__
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            for key, value in defaults.items():
                loaded.setdefault(key, value)
            return loaded
        return defaults.copy()

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)

    def apply_overrides(self, args: argparse.Namespace) -> None:
        if args.cam is not None:
            self.data["cam_index"] = int(args.cam)
        if args.port:
            self.data["midi_port_substr"] = args.port
        if args.ui_scale is not None:
            self.data["ui_scale"] = float(args.ui_scale)

    def reset_preserving(self, keys_to_keep: Iterable[str]) -> None:
        keep = {key: self.data[key] for key in keys_to_keep}
        self.data.clear()
        self.data.update(DEFAULTS.__dict__)
        self.data.update(keep)


def find_midi_port(substr: str) -> str:
    names = get_output_names()
    for name in names:
        if substr.lower() in name.lower():
            return name
    raise SystemExit(f"[ERR] MIDI out not found: '{substr}'. Found: {names}")


def open_camera(idx: int, width: int, height: int, fps: int) -> cv.VideoCapture:
    for backend, name in [
        (cv.CAP_DSHOW, "DSHOW"),
        (cv.CAP_MSMF, "MSMF"),
        (cv.CAP_ANY, "ANY"),
    ]:
        cap = cv.VideoCapture(idx, backend)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv.CAP_PROP_FPS, fps)
            ok, _ = cap.read()
            if ok:
                print(f"[CAM] {name} idx={idx} {int(width)}x{int(height)}@{int(fps)}")
                return cap
        cap.release()
    raise SystemExit(f"[ERR] Camera open failed idx={idx}")


def solve_pose(
    pts2d: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    prev_rvec: Optional[np.ndarray] = None,
    prev_tvec: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    focal = width / (2 * np.tan(np.deg2rad(fov_deg / 2)))
    intrinsic = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32
    )
    dist = np.zeros((4, 1), dtype=np.float32)
    use_guess = prev_rvec is not None and prev_tvec is not None
    ok, rvec, tvec = cv.solvePnP(
        MODEL_POINTS,
        pts2d,
        intrinsic,
        dist,
        rvec=(prev_rvec if use_guess else None),
        tvec=(prev_tvec if use_guess else None),
        useExtrinsicGuess=use_guess,
        flags=cv.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None
    try:
        rvec, tvec = cv.solvePnPRefineVVS(MODEL_POINTS, pts2d, intrinsic, dist, rvec, tvec)
    except Exception:
        pass
    rotation_matrix, _ = cv.Rodrigues(rvec)
    return rotation_matrix, rvec, tvec


def euler_from_R_rel(rot: np.ndarray) -> Tuple[float, float, float]:
    """Relative yaw (Y), pitch (X) and roll (Z) in degrees."""

    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
        yaw = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
        roll = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
        yaw = np.degrees(np.arctan2(-rot[0, 1], rot[1, 1]))
        roll = 0.0
    return yaw, pitch, roll


def roll_from_eyes_2d(landmarks, width: int, height: int) -> float:
    left_x, left_y = landmarks[33].x * width, landmarks[33].y * height
    right_x, right_y = landmarks[263].x * width, landmarks[263].y * height
    return -np.degrees(np.arctan2(right_y - left_y, right_x - left_x))


def unwrap(prev: Optional[Sequence[float]], curr: Sequence[float]) -> Tuple[float, ...]:
    if prev is None:
        return tuple(curr)
    unwrapped = []
    for previous, current in zip(prev, curr):
        delta = current - previous
        if delta > 180:
            current -= 360
        if delta < -180:
            current += 360
        unwrapped.append(current)
    return tuple(unwrapped)


def eff_range(rng: Sequence[float], sens_pct: float) -> List[float]:
    half = max(abs(rng[0]), abs(rng[1]))
    half = max(1.0, half) * (sens_pct / 100.0)
    return [-half, half]


def map_centered(val_deg: float, rng: Sequence[float]) -> int:
    mn, mx = rng
    mx = max(mx, mn + 1e-6)
    norm = (val_deg - mn) / (mx - mn)
    return int(clamp(round(norm * 127), 0, 127))


def map_abs(val_deg: float, rng: Sequence[float]) -> int:
    half = max(abs(rng[0]), abs(rng[1]), 1e-6)
    norm = abs(val_deg) / half
    return int(clamp(round(norm * 127), 0, 127))

class ControlPanel:
    """Simple drawn UI with buttons, readouts and CC meters."""

    def __init__(
        self,
        window: str,
        width: int = 420,
        height: int = 620,
        status_lines: int = 5,
        scale: float = 1.0,
    ) -> None:
        self.win = window
        self.base_w, self.base_h = width, height
        self.scale = scale
        self.w = int(round(self.base_w * self.scale))
        self.h = int(round(self.base_h * self.scale))
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.buttons: List[Tuple[str, str]] = []
        self.button_boxes: List[Tuple[int, int, int, int, str, str]] = []
        self.clicked_action: Optional[str] = None
        self.readouts: List[Tuple[str, str]] = []
        self.centered: Tuple[int, int, int] = (0, 0, 0)
        self.abscc: Tuple[int, int, int] = (0, 0, 0)
        self.status = collections.deque(maxlen=status_lines)
        self.dirty = True
        self.flash_idx: Optional[int] = None
        self.flash_until = 0.0
        cv.setMouseCallback(self.win, self.on_mouse)

    def set_scale(self, scale: float) -> None:
        if abs(scale - self.scale) < 1e-6:
            return
        self.scale = scale
        self.w = int(round(self.base_w * self.scale))
        self.h = int(round(self.base_h * self.scale))
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.button_boxes = []
        self.mark_dirty()

    def _scaled_int(self, value: float) -> int:
        return int(round(value * self.scale))

    def _thickness(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    def set_buttons(self, items: Iterable[Tuple[str, str]]) -> None:
        self.buttons = list(items)
        self.button_boxes = []
        self.mark_dirty()

    def update_readouts(self, items: Iterable[Tuple[str, str]]) -> None:
        self.readouts = list(items)
        self.mark_dirty()

    def update_cc_values(
        self, centered: Sequence[int], abscc: Sequence[int]
    ) -> None:
        self.centered = tuple(centered)  # type: ignore[assignment]
        self.abscc = tuple(abscc)  # type: ignore[assignment]
        self.mark_dirty()

    def notify(self, text: str) -> None:
        self.status.appendleft(text)
        self.mark_dirty()

    def consume_action(self) -> Optional[str]:
        action = self.clicked_action
        self.clicked_action = None
        return action

    def mark_dirty(self) -> None:
        self.dirty = True

    def on_mouse(self, event, x, y, *_args) -> None:
        if event != cv.EVENT_LBUTTONDOWN:
            return
        for idx, (x0, y0, x1, y1, _label, action) in enumerate(self.button_boxes):
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.clicked_action = action
                self.flash_idx = idx
                self.flash_until = time.time() + 0.2
                self.mark_dirty()
                break

    def render_if_needed(self) -> None:
        if self.dirty:
            self.render()

    def render(self) -> None:
        self.dirty = False
        if self.img.shape[0] != self.h or self.img.shape[1] != self.w:
            self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.img[:] = (28, 28, 28)

        font = cv.FONT_HERSHEY_SIMPLEX
        pad = self._scaled_int(20)
        title_y = self._scaled_int(34)
        cv.putText(
            self.img,
            "Control Panel",
            (pad, title_y),
            font,
            0.8 * self.scale,
            (250, 250, 250),
            self._thickness(2),
            cv.LINE_AA,
        )

        y = self._scaled_int(64)
        for label, value in self.readouts:
            cv.putText(
                self.img,
                f"{label}: {value}",
                (pad, y),
                font,
                0.55 * self.scale,
                (220, 220, 220),
                self._thickness(1),
                cv.LINE_AA,
            )
            y += self._scaled_int(24)

        y += self._scaled_int(6)
        cv.putText(
            self.img,
            "CC output",
            (pad, y),
            font,
            0.6 * self.scale,
            (200, 210, 255),
            self._thickness(1),
            cv.LINE_AA,
        )
        y += self._scaled_int(24)
        cv.putText(
            self.img,
            f"Centered CC1/11/74  {self.centered[0]:3d}  {self.centered[1]:3d}  {self.centered[2]:3d}",
            (pad, y),
            font,
            0.5 * self.scale,
            (210, 210, 210),
            self._thickness(1),
            cv.LINE_AA,
        )
        y += self._scaled_int(20)
        cv.putText(
            self.img,
            f"Absolute CC21/22/23 {self.abscc[0]:3d}  {self.abscc[1]:3d}  {self.abscc[2]:3d}",
            (pad, y),
            font,
            0.5 * self.scale,
            (210, 210, 210),
            self._thickness(1),
            cv.LINE_AA,
        )

        y += self._scaled_int(32)
        cv.putText(
            self.img,
            "Actions",
            (pad, y),
            font,
            0.6 * self.scale,
            (200, 210, 255),
            self._thickness(1),
            cv.LINE_AA,
        )
        y += self._scaled_int(10)

        now = time.time()
        btn_width = self.w - 2 * pad
        btn_height = self._scaled_int(36)
        gap = self._scaled_int(10)
        x0 = pad
        start_y = y + self._scaled_int(4)
        self.button_boxes = []
        for idx, (label, action) in enumerate(self.buttons):
            y0 = start_y + idx * (btn_height + gap)
            y1 = y0 + btn_height
            x1 = x0 + btn_width
            self.button_boxes.append((x0, y0, x1, y1, label, action))
            cv.rectangle(
                self.img,
                (x0, y0),
                (x1, y1),
                (200, 200, 200),
                self._thickness(1),
            )
            if idx == self.flash_idx and now < self.flash_until:
                cv.rectangle(
                    self.img,
                    (x0, y0),
                    (x1, y1),
                    (0, 255, 0),
                    self._thickness(2),
                )
            cv.putText(
                self.img,
                label,
                (x0 + self._scaled_int(10), y0 + self._scaled_int(24)),
                font,
                0.55 * self.scale,
                (230, 230, 230),
                self._thickness(1),
                cv.LINE_AA,
            )

        y = start_y + len(self.buttons) * (btn_height + gap) + self._scaled_int(16)
        if y > self.h - self._scaled_int(90):
            y = self.h - self._scaled_int(90)
        cv.putText(
            self.img,
            "Status",
            (pad, y),
            font,
            0.6 * self.scale,
            (200, 210, 255),
            self._thickness(1),
            cv.LINE_AA,
        )
        y += self._scaled_int(24)
        for line in self.status:
            cv.putText(
                self.img,
                line,
                (pad, y),
                font,
                0.5 * self.scale,
                (210, 210, 210),
                self._thickness(1),
                cv.LINE_AA,
            )
            y += self._scaled_int(18)

        cv.imshow(self.win, self.img)


def draw_bars(
    frame: np.ndarray,
    triplet: Sequence[int],
    title: str,
    origin: Tuple[int, int] = (20, 80),
    scale: float = 1.0,
) -> None:
    font = cv.FONT_HERSHEY_SIMPLEX
    x0 = int(round(origin[0] * scale))
    y0 = int(round(origin[1] * scale))
    width = int(round(360 * scale))
    height = max(1, int(round(20 * scale)))
    gap = max(1, int(round(10 * scale)))
    thickness = max(1, int(round(1 * scale)))
    label_font_scale = 0.55 * scale
    cv.putText(
        frame,
        title,
        (x0, y0 - max(1, int(round(12 * scale)))),
        font,
        0.6 * scale,
        (200, 200, 200),
        thickness,
        cv.LINE_AA,
    )
    labels = [("Yaw", triplet[0]), ("Pitch", triplet[1]), ("Roll", triplet[2])]
    for idx, (name, value) in enumerate(labels):
        y = y0 + idx * (height + gap)
        cv.rectangle(frame, (x0, y), (x0 + width, y + height), (160, 160, 160), thickness)
        fill = int((value / 127.0) * width)
        cv.rectangle(frame, (x0, y), (x0 + fill, y + height), (0, 255, 0), -1)
        cv.putText(
            frame,
            f"{name} {value:3d}",
            (x0 + width + max(1, int(round(10 * scale))), y + height - max(1, int(round(4 * scale)))),
            font,
            label_font_scale,
            (255, 255, 255),
            thickness,
            cv.LINE_AA,
        )

TB_YAW = "Yaw span (deg)"
TB_PITCH = "Pitch span (deg)"
TB_ROLL = "Roll span (deg)"
TB_GLOBAL = "Sensitivity (%)"
TB_DEADZONE = "Deadzone (0.1deg)"
TB_SMOOTH = "Smoothing (x100)"
TB_RATE = "Send rate (Hz)"
TB_SEND = "Send on/off"
TB_CHAN = "MIDI channel"

BAR_BLOCK_BASE_OFFSET = 3 * (20 + 10) + 26

@dataclass
class PoseResult:
    ok: bool
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    pts2d: Optional[np.ndarray] = None
    landmarks: Optional[landmark_pb2.NormalizedLandmarkList] = None


@dataclass
class PoseState:
    prev_rvec: Optional[np.ndarray] = None
    prev_tvec: Optional[np.ndarray] = None
    baseline: Optional[np.ndarray] = None
    prev_euler: Optional[Tuple[float, float, float]] = None
    yaw_s: Optional[float] = None
    pitch_s: Optional[float] = None
    roll_s: Optional[float] = None
    last_rotation: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_rvec = None
        self.prev_tvec = None
        self.baseline = None
        self.prev_euler = None
        self.yaw_s = None
        self.pitch_s = None
        self.roll_s = None
        self.last_rotation = None


class PoseEstimator:
    def __init__(self, fov_deg: float) -> None:
        self.fov_deg = fov_deg
        self.state = PoseState()

    def set_fov(self, fov_deg: float) -> None:
        self.fov_deg = fov_deg

    def reset(self) -> None:
        self.state.reset()

    def calibrate(self) -> bool:
        if self.state.last_rotation is None:
            return False
        self.state.baseline = self.state.last_rotation.copy()
        self.state.prev_euler = (0.0, 0.0, 0.0)
        self.state.yaw_s = self.state.pitch_s = self.state.roll_s = 0.0
        return True

    def process(
        self,
        face_landmarks: Optional[Sequence[landmark_pb2.NormalizedLandmarkList]],
        width: int,
        height: int,
        cfg: dict,
    ) -> PoseResult:
        if not face_landmarks:
            return PoseResult(False)

        landmarks = face_landmarks[0]
        lm_seq = landmarks.landmark
        pts2d = np.array([(lm_seq[i].x * width, lm_seq[i].y * height) for i in LM], dtype=np.float32)
        rotation, self.state.prev_rvec, self.state.prev_tvec = solve_pose(
            pts2d, width, height, self.fov_deg, self.state.prev_rvec, self.state.prev_tvec
        )
        if rotation is None:
            return PoseResult(False, pts2d=pts2d, landmarks=landmarks)

        self.state.last_rotation = rotation
        if self.state.baseline is None:
            self.state.baseline = rotation.copy()

        relative = self.state.baseline.T @ rotation
        yaw, pitch, roll = euler_from_R_rel(relative)
        roll = 0.7 * roll + 0.3 * roll_from_eyes_2d(lm_seq, width, height)
        yaw, pitch, roll = unwrap(self.state.prev_euler, (yaw, pitch, roll))
        self.state.prev_euler = (yaw, pitch, roll)

        if cfg["invert_yaw"]:
            yaw = -yaw
        if cfg["invert_pitch"]:
            pitch = -pitch
        if cfg["invert_roll"]:
            roll = -roll

        alpha = cfg["smooth_alpha"]
        self.state.yaw_s = ema(self.state.yaw_s, yaw, alpha)
        self.state.pitch_s = ema(self.state.pitch_s, pitch, alpha)
        self.state.roll_s = ema(self.state.roll_s, roll, alpha)

        return PoseResult(
            True,
            yaw=self.state.yaw_s,
            pitch=self.state.pitch_s,
            roll=self.state.roll_s,
            pts2d=pts2d,
            landmarks=landmarks,
        )


class MidiController:
    def __init__(self, port_substr: str, channel: int) -> None:
        port_name = find_midi_port(port_substr)
        self.port = open_output(port_name)
        print(f"[MIDI] {port_name}")
        self.channel = int(clamp(channel, 0, 15))

    def close(self) -> None:
        self.port.close()

    def set_channel(self, channel: int) -> None:
        self.channel = int(clamp(channel, 0, 15))

    def send_pose_values(
        self,
        cfg: dict,
        centered: Sequence[int],
        absolute: Sequence[int],
    ) -> None:
        for control, value in zip(
            (cfg["cc_yaw"], cfg["cc_pitch"], cfg["cc_roll"]),
            centered,
        ):
            self.port.send(
                Message("control_change", channel=self.channel, control=int(control), value=int(value))
            )
        for control, value in zip(
            (cfg["cc_yaw_abs"], cfg["cc_pitch_abs"], cfg["cc_roll_abs"]),
            absolute,
        ):
            self.port.send(
                Message("control_change", channel=self.channel, control=int(control), value=int(value))
            )

    def send_learn_pulse(self, cc: int) -> None:
        self.port.send(Message("control_change", channel=self.channel, control=cc, value=127))
        self.port.send(Message("control_change", channel=self.channel, control=cc, value=0))


class HeadCCApp:
    def __init__(self, cfg_mgr: ConfigManager) -> None:
        self.cfg_mgr = cfg_mgr
        self.cfg = cfg_mgr.data
        self.ui_scale = clamp(float(self.cfg.get("ui_scale", 1.0)), UI_SCALE_MIN, UI_SCALE_MAX)
        self.cfg["ui_scale"] = self.ui_scale

        self.midi = MidiController(self.cfg["midi_port_substr"], self.cfg["midi_channel"])
        self.cap = open_camera(
            self.cfg["cam_index"], self.cfg["width"], self.cfg["height"], self.cfg["fps"]
        )

        self.pose = PoseEstimator(self.cfg["fov_deg"])

        self.face_mesh = mp_solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.draw_utils = mp_solutions.drawing_utils
        self.draw_styles = mp_solutions.drawing_styles
        self.mesh_spec = self.draw_styles.get_default_face_mesh_tesselation_style()
        self.contour_spec = self.draw_styles.get_default_face_mesh_contours_style()
        self.iris_spec = self.draw_styles.get_default_face_mesh_iris_connections_style()

        self.main_window = "Head -> MIDI CC"
        self.ctrl_window = "Control Panel"
        self.panel = ControlPanel(self.ctrl_window, 420, 640, scale=self.ui_scale)

        self.centered_cc: Tuple[int, int, int] = (64, 64, 64)
        self.abs_cc: Tuple[int, int, int] = (0, 0, 0)
        self.last_send = 0.0

        self._setup_windows()
        self._setup_trackbars()
        self._sync_trackbars()
        self._init_panel()

    def _setup_windows(self) -> None:
        cv.namedWindow(self.main_window, cv.WINDOW_NORMAL)
        cv.moveWindow(self.main_window, 40, 40)
        cv.resizeWindow(
            self.main_window,
            int(round(1050 * self.ui_scale)),
            int(round(700 * self.ui_scale)),
        )
        cv.namedWindow(self.ctrl_window, cv.WINDOW_NORMAL)
        cv.moveWindow(self.ctrl_window, 1120, 40)
        cv.resizeWindow(self.ctrl_window, self.panel.w, self.panel.h)

    def _setup_trackbars(self) -> None:
        def mk_tb(name: str, init: int, maxv: int) -> None:
            cv.createTrackbar(name, self.ctrl_window, init, maxv, lambda *_: None)

        def init_span_tb(label: str, key: str, default_span: int) -> None:
            rng = self.cfg[key]
            span = int(round(abs(rng[1] - rng[0]))) or default_span
            span = int(clamp(span, 10, 180))
            mk_tb(label, span, 180)

        init_span_tb(TB_YAW, "yaw_range_deg", 90)
        init_span_tb(TB_PITCH, "pitch_range_deg", 60)
        init_span_tb(TB_ROLL, "roll_range_deg", 70)
        mk_tb(TB_GLOBAL, int(self.cfg["global_sens_pct"]), 300)
        mk_tb(TB_DEADZONE, int(self.cfg["deadzone_deg"] * 10), 100)
        mk_tb(TB_SMOOTH, int(self.cfg["smooth_alpha"] * 100), 100)
        mk_tb(TB_RATE, int(self.cfg["send_rate_hz"]), 120)
        mk_tb(TB_SEND, 1, 1)
        mk_tb(TB_CHAN, int(self.cfg["midi_channel"] + 1), 16)

    def _sync_trackbars(self) -> None:
        def total_span(rng: Sequence[float]) -> int:
            return int(round(abs(rng[1] - rng[0])))

        cv.setTrackbarPos(TB_YAW, self.ctrl_window, total_span(self.cfg["yaw_range_deg"]))
        cv.setTrackbarPos(TB_PITCH, self.ctrl_window, total_span(self.cfg["pitch_range_deg"]))
        cv.setTrackbarPos(TB_ROLL, self.ctrl_window, total_span(self.cfg["roll_range_deg"]))
        cv.setTrackbarPos(TB_DEADZONE, self.ctrl_window, int(self.cfg["deadzone_deg"] * 10))
        cv.setTrackbarPos(TB_SMOOTH, self.ctrl_window, int(self.cfg["smooth_alpha"] * 100))
        cv.setTrackbarPos(TB_RATE, self.ctrl_window, int(self.cfg["send_rate_hz"]))
        cv.setTrackbarPos(TB_GLOBAL, self.ctrl_window, int(self.cfg["global_sens_pct"]))
        cv.setTrackbarPos(TB_CHAN, self.ctrl_window, int(self.cfg["midi_channel"] + 1))
        cv.setTrackbarPos(TB_SEND, self.ctrl_window, 1)

    def _init_panel(self) -> None:
        rows = [
            ("Send CC1 (Yaw)", f"cc:{self.cfg['cc_yaw']}"),
            ("Send CC11 (Pitch)", f"cc:{self.cfg['cc_pitch']}"),
            ("Send CC74 (Roll)", f"cc:{self.cfg['cc_roll']}"),
            ("Send CC21 (YawAbs)", f"cc:{self.cfg['cc_yaw_abs']}"),
            ("Send CC22 (PitchAbs)", f"cc:{self.cfg['cc_pitch_abs']}"),
            ("Send CC23 (RollAbs)", f"cc:{self.cfg['cc_roll_abs']}"),
            ("Save settings", "save"),
            ("Reset settings", "reset"),
        ]
        self.panel.set_buttons(rows)
        self.panel.notify("Controls ready. Space toggles send, Q quits.")
        self.panel.notify("Press C to recalibrate origin; Y/P/R invert axes.")
        self.panel.notify("Use [ / ] or - / + to change UI scale.")
        self.panel.render_if_needed()

    def _apply_ui_scale(self, new_scale: float) -> bool:
        new_scale = clamp(float(new_scale), UI_SCALE_MIN, UI_SCALE_MAX)
        if abs(new_scale - self.ui_scale) < 1e-6:
            return False
        self.ui_scale = new_scale
        self.cfg["ui_scale"] = self.ui_scale
        cv.resizeWindow(
            self.main_window,
            int(round(1050 * self.ui_scale)),
            int(round(700 * self.ui_scale)),
        )
        self.panel.set_scale(self.ui_scale)
        cv.resizeWindow(self.ctrl_window, self.panel.w, self.panel.h)
        self.panel.render_if_needed()
        return True

    def _read_controls(self) -> bool:
        def span_from_tb(name: str) -> float:
            return float(clamp(cv.getTrackbarPos(name, self.ctrl_window), 10, 180))

        base_yaw_span = span_from_tb(TB_YAW)
        base_pitch_span = span_from_tb(TB_PITCH)
        base_roll_span = span_from_tb(TB_ROLL)
        self.cfg["global_sens_pct"] = clamp(cv.getTrackbarPos(TB_GLOBAL, self.ctrl_window), 10, 300)
        self.cfg["deadzone_deg"] = cv.getTrackbarPos(TB_DEADZONE, self.ctrl_window) / 10.0
        self.cfg["smooth_alpha"] = cv.getTrackbarPos(TB_SMOOTH, self.ctrl_window) / 100.0
        self.cfg["send_rate_hz"] = max(1, cv.getTrackbarPos(TB_RATE, self.ctrl_window) or 30)
        send_enabled = cv.getTrackbarPos(TB_SEND, self.ctrl_window) == 1
        self.cfg["midi_channel"] = int(clamp(cv.getTrackbarPos(TB_CHAN, self.ctrl_window) - 1, 0, 15))
        self.midi.set_channel(self.cfg["midi_channel"])

        self.cfg["yaw_range_deg"] = [-base_yaw_span / 2.0, base_yaw_span / 2.0]
        self.cfg["pitch_range_deg"] = [-base_pitch_span / 2.0, base_pitch_span / 2.0]
        self.cfg["roll_range_deg"] = [-base_roll_span / 2.0, base_roll_span / 2.0]

        return send_enabled

    def _update_midi(
        self,
        pose: PoseResult,
        send_enabled: bool,
        yaw_rng: Sequence[float],
        pitch_rng: Sequence[float],
        roll_rng: Sequence[float],
    ) -> None:
        now = time.time()
        if not (
            pose.ok
            and send_enabled
            and (now - self.last_send) >= 1.0 / self.cfg["send_rate_hz"]
        ):
            return

        self.last_send = now

        def apply_deadzone(value: float) -> float:
            return 0.0 if abs(value) < self.cfg["deadzone_deg"] else value

        yaw_v = apply_deadzone(pose.yaw)
        pitch_v = apply_deadzone(pose.pitch)
        roll_v = apply_deadzone(pose.roll)

        cy = map_centered(yaw_v, yaw_rng)
        cp = map_centered(pitch_v, pitch_rng)
        cr = map_centered(roll_v, roll_rng)
        self.centered_cc = (cy, cp, cr)

        ay = map_abs(yaw_v, yaw_rng)
        ap = map_abs(pitch_v, pitch_rng)
        ar = map_abs(roll_v, roll_rng)
        self.abs_cc = (ay, ap, ar)

        self.midi.send_pose_values(self.cfg, self.centered_cc, self.abs_cc)

    def _draw_face(self, frame: np.ndarray, pose: PoseResult) -> None:
        if not pose.landmarks:
            return
        self.draw_utils.draw_landmarks(
            frame,
            pose.landmarks,
            mp_solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mesh_spec,
        )
        self.draw_utils.draw_landmarks(
            frame,
            pose.landmarks,
            mp_solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.contour_spec,
        )
        self.draw_utils.draw_landmarks(
            frame,
            pose.landmarks,
            mp_solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.iris_spec,
        )

    def _render_frame(
        self,
        frame: np.ndarray,
        pose: PoseResult,
        yaw_rng: Sequence[float],
        pitch_rng: Sequence[float],
        roll_rng: Sequence[float],
        send_enabled: bool,
    ) -> None:
        height, width = frame.shape[:2]
        if pose.ok:
            cv.putText(
                frame,
                f"Yaw {pose.yaw:6.1f}  Pitch {pose.pitch:6.1f}  Roll {pose.roll:6.1f}",
                (int(round(20 * self.ui_scale)), int(round(40 * self.ui_scale))),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0 * self.ui_scale,
                (0, 255, 0),
                max(1, int(round(2 * self.ui_scale))),
                cv.LINE_AA,
            )
            if pose.pts2d is not None:
                for x, y in pose.pts2d.astype(int):
                    cv.circle(frame, (x, y), 3, (0, 255, 0), -1)
        else:
            cv.putText(
                frame,
                "No face",
                (int(round(20 * self.ui_scale)), int(round(40 * self.ui_scale))),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0 * self.ui_scale,
                (0, 0, 255),
                max(1, int(round(2 * self.ui_scale))),
                cv.LINE_AA,
            )

        self._draw_face(frame, pose)

        draw_bars(
            frame,
            self.centered_cc,
            "Centered CC  CC1/11/74",
            origin=(20, 80),
            scale=self.ui_scale,
        )
        draw_bars(
            frame,
            self.abs_cc,
            "ABS CC       CC21/22/23",
            origin=(20, 80 + BAR_BLOCK_BASE_OFFSET),
            scale=self.ui_scale,
        )

        footer = (
            f"Send={int(send_enabled)} | Ch={self.cfg['midi_channel'] + 1} | rate {self.cfg['send_rate_hz']}Hz | "
            f"dz {self.cfg['deadzone_deg']:.1f}° | smooth {self.cfg['smooth_alpha']:.2f} | sens {self.cfg['global_sens_pct']}% | "
            f"UI {self.ui_scale:.2f}x"
        )
        cv.putText(
            frame,
            footer,
            (int(round(20 * self.ui_scale)), height - int(round(20 * self.ui_scale))),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6 * self.ui_scale,
            (200, 200, 200),
            max(1, int(round(1 * self.ui_scale))),
            cv.LINE_AA,
        )

        cv.imshow(self.main_window, frame)

        yaw_half = max(abs(yaw_rng[0]), abs(yaw_rng[1]))
        pitch_half = max(abs(pitch_rng[0]), abs(pitch_rng[1]))
        roll_half = max(abs(roll_rng[0]), abs(roll_rng[1]))
        status = "Tracking" if pose.ok else "No face"
        send_state = "On" if send_enabled else "Muted"
        self.panel.update_readouts(
            [
                ("Yaw range", f"±{yaw_half:.0f}°"),
                ("Pitch range", f"±{pitch_half:.0f}°"),
                ("Roll range", f"±{roll_half:.0f}°"),
                ("Sensitivity", f"{self.cfg['global_sens_pct']}%"),
                ("Deadzone", f"{self.cfg['deadzone_deg']:.1f}°"),
                ("Smooth", f"{self.cfg['smooth_alpha']:.2f}"),
                ("Rate", f"{self.cfg['send_rate_hz']} Hz"),
                ("Channel", f"{self.cfg['midi_channel'] + 1}"),
                ("Send", send_state),
                ("UI scale", f"{self.ui_scale:.2f}x"),
                ("Face", status),
            ]
        )
        self.panel.update_cc_values(self.centered_cc, self.abs_cc)
        self.panel.render_if_needed()

    def _handle_actions(self) -> None:
        action = self.panel.consume_action()
        if not action:
            return
        if action.startswith("cc:"):
            cc = int(action.split(":")[1])
            self.midi.send_learn_pulse(cc)
            self.panel.notify(f"Learn pulse sent on CC{cc} (ch {self.cfg['midi_channel'] + 1})")
        elif action == "save":
            self.cfg_mgr.save()
            self.panel.notify("Settings saved")
        elif action == "reset":
            self.cfg_mgr.reset_preserving(["midi_port_substr", "cam_index"])
            self.cfg = self.cfg_mgr.data
            self.pose.reset()
            self.pose.set_fov(self.cfg["fov_deg"])
            self.centered_cc = (64, 64, 64)
            self.abs_cc = (0, 0, 0)
            self._sync_trackbars()
            self._apply_ui_scale(self.cfg["ui_scale"])
            self.midi.set_channel(self.cfg["midi_channel"])
            self.cfg_mgr.save()
            self.panel.notify("Settings reset to defaults")
            self.panel.mark_dirty()

    def _handle_keypress(self, pose: PoseResult, send_enabled: bool) -> bool:
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return True
        if key == ord("c") and pose.ok:
            if self.pose.calibrate():
                self.panel.notify("Origin calibrated")
        elif key == ord(" "):
            cv.setTrackbarPos(TB_SEND, self.ctrl_window, 0 if send_enabled else 1)
            self.panel.notify("Send muted" if send_enabled else "Send enabled")
        elif key == ord("y"):
            self.cfg["invert_yaw"] = not self.cfg["invert_yaw"]
            self.panel.notify(f"Invert yaw: {self.cfg['invert_yaw']}")
        elif key == ord("p"):
            self.cfg["invert_pitch"] = not self.cfg["invert_pitch"]
            self.panel.notify(f"Invert pitch: {self.cfg['invert_pitch']}")
        elif key == ord("r"):
            self.cfg["invert_roll"] = not self.cfg["invert_roll"]
            self.panel.notify(f"Invert roll: {self.cfg['invert_roll']}")
        elif key == ord("s"):
            self.cfg_mgr.save()
            self.panel.notify("Settings saved")
        elif key in (ord("["), ord("{"), ord("-"), ord("_")):
            if self._apply_ui_scale(self.ui_scale - 0.1):
                self.panel.notify(f"UI scale: {self.ui_scale:.2f}x")
        elif key in (ord("]"), ord("}"), ord("+"), ord("=")):
            if self._apply_ui_scale(self.ui_scale + 0.1):
                self.panel.notify(f"UI scale: {self.ui_scale:.2f}x")
        return False

    def run(self) -> None:
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("[ERR] frame read")
                    break
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)
                send_enabled = self._read_controls()
                yaw_rng = eff_range(self.cfg["yaw_range_deg"], self.cfg["global_sens_pct"])
                pitch_rng = eff_range(self.cfg["pitch_range_deg"], self.cfg["global_sens_pct"])
                roll_rng = eff_range(self.cfg["roll_range_deg"], self.cfg["global_sens_pct"])
                pose = self.pose.process(
                    results.multi_face_landmarks,
                    frame.shape[1],
                    frame.shape[0],
                    self.cfg,
                )
                self._update_midi(pose, send_enabled, yaw_rng, pitch_rng, roll_rng)
                self._render_frame(frame, pose, yaw_rng, pitch_rng, roll_rng, send_enabled)
                self._handle_actions()
                if self._handle_keypress(pose, send_enabled):
                    break
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.cfg_mgr.save()
        self.cap.release()
        cv.destroyAllWindows()
        self.midi.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int)
    parser.add_argument("--port", type=str)
    parser.add_argument("--showports", action="store_true")
    parser.add_argument("--ui-scale", type=float)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.showports:
        print(get_output_names())
        return

    cfg_mgr = ConfigManager()
    cfg_mgr.apply_overrides(args)
    app = HeadCCApp(cfg_mgr)
    app.run()


if __name__ == "__main__":
    main()
