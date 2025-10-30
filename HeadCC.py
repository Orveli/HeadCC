# HeadCC.py — robust baseline, 6x CC, mute toggle, global sensitivity, control panel, save/reset
# Python 3.10 x64, mediapipe 0.10.x
import os, json, time, argparse, collections
from dataclasses import dataclass
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe import solutions as mp_solutions
from mido import Message, open_output, get_output_names

CONFIG_PATH = "headcc_cfg.json"
DEFAULTS = {
    "midi_port_substr": "HeadCC",
    "midi_channel": 0,  # 0..15
    # Centered CCs
    "cc_yaw": 1, "cc_pitch": 11, "cc_roll": 74,
    # ABS CCs
    "cc_yaw_abs": 21, "cc_pitch_abs": 22, "cc_roll_abs": 23,
    # Runtime
    "send_rate_hz": 30,
    "smooth_alpha": 0.25,
    "deadzone_deg": 2.0,
    # base symmetric spans (±deg). Global sensitivity scales these.
    "yaw_range_deg":   [-45.0, 45.0],
    "pitch_range_deg": [-30.0, 30.0],
    "roll_range_deg":  [-35.0, 35.0],
    "global_sens_pct": 100,     # 10..300 %
    "invert_yaw": False, "invert_pitch": False, "invert_roll": False,
    # camera
    "cam_index": 0, "width": 1280, "height": 720, "fps": 60, "fov_deg": 60.0,
    # UI
    "ui_scale": 1.0,
}

UI_SCALE_MIN = 0.6
UI_SCALE_MAX = 2.5

LM = [1, 152, 33, 263, 61, 291]  # nose, chin, eye outer L/R, mouth L/R
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
], dtype=np.float32)

# ---------- utils ----------
def load_cfg():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k,v in DEFAULTS.items(): d.setdefault(k, v)
        return d
    return DEFAULTS.copy()

def save_cfg(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(cfg, f, indent=2)

def clamp(v,a,b): return max(a,min(b,v))
def ema(prev, new, a): return new if prev is None else (a*new + (1-a)*prev)

def find_midi_port(substr):
    names = get_output_names()
    for n in names:
        if substr.lower() in n.lower(): return n
    raise SystemExit(f"[ERR] MIDI out not found: '{substr}'. Found: {names}")

def open_camera(idx, w, h, fps):
    for backend,name in [(cv.CAP_DSHOW,"DSHOW"), (cv.CAP_MSMF,"MSMF"), (cv.CAP_ANY,"ANY")]:
        cap = cv.VideoCapture(idx, backend)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv.CAP_PROP_FPS, fps)
            ok,_ = cap.read()
            if ok:
                print(f"[CAM] {name} idx={idx} {int(w)}x{int(h)}@{int(fps)}")
                return cap
        cap.release()
    raise SystemExit(f"[ERR] Camera open failed idx={idx}")

# ---------- pose ----------
def solve_pose(pts2d, w, h, fov_deg, prev_rvec=None, prev_tvec=None):
    focal = w / (2*np.tan(np.deg2rad(fov_deg/2)))
    K = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float32)
    dist = np.zeros((4,1), dtype=np.float32)
    use_guess = prev_rvec is not None and prev_tvec is not None
    ok, rvec, tvec = cv.solvePnP(MODEL_POINTS, pts2d, K, dist,
                                 rvec=(prev_rvec if use_guess else None),
                                 tvec=(prev_tvec if use_guess else None),
                                 useExtrinsicGuess=use_guess,
                                 flags=cv.SOLVEPNP_ITERATIVE)
    if not ok: return None, None, None
    try:
        rvec, tvec = cv.solvePnPRefineVVS(MODEL_POINTS, pts2d, K, dist, rvec, tvec)
    except Exception:
        pass
    R,_ = cv.Rodrigues(rvec)
    return R, rvec, tvec

def euler_from_R_rel(R):
    # Relative yaw(Y), pitch(X), roll(Z). Degrees.
    sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
        roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))
    else:
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw   = np.degrees(np.arctan2(-R[0,1], R[1,1]))
        roll  = 0.0
    return yaw, pitch, roll

def roll_from_eyes_2d(lms, w, h):
    xL, yL = lms[33].x*w,  lms[33].y*h
    xR, yR = lms[263].x*w, lms[263].y*h
    return -np.degrees(np.arctan2(yR - yL, xR - xL))

def unwrap(prev, curr):
    if prev is None: return curr
    out=[]
    for p,c in zip(prev, curr):
        d = c - p
        if d > 180:  c -= 360
        if d < -180: c += 360
        out.append(c)
    return tuple(out)

# ---------- mapping ----------
def eff_range(rng, sens_pct):
    half = max(abs(rng[0]), abs(rng[1]))
    half = max(1.0, half) * (sens_pct/100.0)
    return [-half, half]

def map_centered(val_deg, rng):
    mn,mx=rng; mx=max(mx,mn+1e-6)
    norm=(val_deg-mn)/(mx-mn)
    return int(clamp(round(norm*127),0,127))

def map_abs(val_deg, rng):
    half=max(abs(rng[0]),abs(rng[1]),1e-6)
    norm=abs(val_deg)/half
    return int(clamp(round(norm*127),0,127))


# ---------- UI helpers ----------
@dataclass
class SliderState:
    key: str
    label: str
    min_value: float
    max_value: float
    step: float
    value: float
    fmt: str
    cast: type = float

    def __post_init__(self):
        self.set_value(self.value)

    def set_value(self, value):
        value = max(self.min_value, min(self.max_value, value))
        if self.step:
            steps = round((value - self.min_value) / self.step)
            value = self.min_value + steps * self.step
            value = max(self.min_value, min(self.max_value, value))
        self.value = float(value)
        return self.value

    def normalized(self):
        if self.max_value <= self.min_value:
            return 0.0
        return (self.value - self.min_value) / (self.max_value - self.min_value)

    def get_value(self):
        if self.cast is int:
            return int(round(self.value))
        if callable(self.cast) and self.cast not in (int, float):
            return self.cast(self.value)
        return float(round(self.value, 6))

    def formatted(self):
        return self.fmt.format(self.get_value())


@dataclass
class ToggleState:
    key: str
    label: str
    value: bool = False


class ControlPanel:
    def __init__(self, window, w=520, h=780, status_lines=6, scale=1.0):
        self.win = window
        self.base_w, self.base_h = w, h
        self.scale = scale
        self.w = int(round(self.base_w * self.scale))
        self.h = int(round(self.base_h * self.scale))
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.buttons = []
        self.button_boxes = []
        self.clicked_action = None
        self.readouts = []
        self.centered = (0, 0, 0)
        self.abscc = (0, 0, 0)
        self.status = collections.deque(maxlen=status_lines)
        self.dirty = True
        self.flash_idx = None
        self.flash_until = 0.0
        self.sliders = []
        self.slider_map = {}
        self.slider_boxes = []
        self.drag_slider = None
        self.toggles = []
        self.toggle_map = {}
        self.toggle_boxes = []
        cv.setMouseCallback(self.win, self.on_mouse)

    def set_scale(self, scale):
        if abs(scale - self.scale) < 1e-6:
            return
        self.scale = scale
        self.w = int(round(self.base_w * self.scale))
        self.h = int(round(self.base_h * self.scale))
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.slider_boxes = []
        self.toggle_boxes = []
        self.button_boxes = []
        self.mark_dirty()

    def _si(self, value):
        return int(round(value * self.scale))

    def _thickness(self, value):
        return max(1, int(round(value * self.scale)))

    def set_buttons(self, items):
        self.buttons = list(items)
        self.button_boxes = []
        self.mark_dirty()

    def set_sliders(self, specs):
        self.sliders = []
        self.slider_map = {}
        for spec in specs:
            slider = SliderState(
                key=spec["key"],
                label=spec["label"],
                min_value=spec["min"],
                max_value=spec["max"],
                step=spec.get("step", 1.0),
                value=spec.get("value", spec["min"]),
                fmt=spec.get("fmt", "{:.0f}"),
                cast=spec.get("cast", float),
            )
            self.sliders.append(slider)
            self.slider_map[slider.key] = slider
        self.slider_boxes = []
        self.mark_dirty()

    def set_toggles(self, specs):
        self.toggles = []
        self.toggle_map = {}
        for spec in specs:
            toggle = ToggleState(spec["key"], spec["label"], bool(spec.get("value", False)))
            self.toggles.append(toggle)
            self.toggle_map[toggle.key] = toggle
        self.toggle_boxes = []
        self.mark_dirty()

    def set_slider_value(self, key, value):
        slider = self.slider_map.get(key)
        if slider is None:
            return
        prev = slider.value
        slider.set_value(value)
        if abs(prev - slider.value) > 1e-6:
            self.mark_dirty()

    def get_slider_value(self, key):
        slider = self.slider_map.get(key)
        if slider is None:
            return None
        return slider.get_value()

    def set_toggle_state(self, key, value):
        toggle = self.toggle_map.get(key)
        if toggle is None:
            return
        value = bool(value)
        if toggle.value != value:
            toggle.value = value
            self.mark_dirty()

    def get_toggle_state(self, key):
        toggle = self.toggle_map.get(key)
        return bool(toggle.value) if toggle else False

    def toggle_state(self, key):
        toggle = self.toggle_map.get(key)
        if toggle is None:
            return False
        toggle.value = not toggle.value
        self.mark_dirty()
        return toggle.value

    def update_readouts(self, items):
        self.readouts = list(items)
        self.mark_dirty()

    def update_cc_values(self, centered, abscc):
        self.centered = tuple(centered)
        self.abscc = tuple(abscc)
        self.mark_dirty()

    def notify(self, text):
        self.status.appendleft(text)
        self.mark_dirty()

    def consume_action(self):
        act = self.clicked_action
        self.clicked_action = None
        return act

    def mark_dirty(self):
        self.dirty = True

    def _find_slider_box(self, key):
        for box in self.slider_boxes:
            if box["key"] == key:
                return box
        return None

    def _update_slider_from_pos(self, box, x):
        slider = self.slider_map.get(box["key"])
        if slider is None:
            return
        span = box["x1"] - box["x0"]
        if span <= 0:
            return
        ratio = (x - box["x0"]) / span
        ratio = clamp(ratio, 0.0, 1.0)
        value = slider.min_value + ratio * (slider.max_value - slider.min_value)
        prev = slider.value
        slider.set_value(value)
        if abs(prev - slider.value) > 1e-6:
            self.mark_dirty()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            for box in self.button_boxes:
                if box["x0"] <= x <= box["x1"] and box["y0"] <= y <= box["y1"]:
                    self.clicked_action = box["action"]
                    self.flash_idx = box["idx"]
                    self.flash_until = time.time() + 0.25
                    self.mark_dirty()
                    return
            for box in self.toggle_boxes:
                if box["x0"] <= x <= box["x1"] and box["y0"] <= y <= box["y1"]:
                    toggle = self.toggle_map.get(box["key"])
                    if toggle:
                        toggle.value = not toggle.value
                        self.mark_dirty()
                    return
            for box in self.slider_boxes:
                if box["x0"] <= x <= box["x1"] and box["y0"] <= y <= box["y1"]:
                    self.drag_slider = box["key"]
                    self._update_slider_from_pos(box, x)
                    return
        elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON) and self.drag_slider:
            box = self._find_slider_box(self.drag_slider)
            if box:
                self._update_slider_from_pos(box, x)
        elif event == cv.EVENT_LBUTTONUP:
            self.drag_slider = None

    def render_if_needed(self):
        if self.dirty:
            self.render()

    def _draw_section_title(self, y, text, font, color, pad):
        cv.putText(self.img, text, (pad, y), font, 0.62 * self.scale, color,
                   self._thickness(1) + 1, cv.LINE_AA)
        return y + self._si(28)

    def _draw_toggles(self, y, font, pad, text_color):
        self.toggle_boxes = []
        if not self.toggles:
            return y
        gap = self._si(12)
        toggle_h = self._si(42)
        cols = 2
        available = self.w - 2 * pad - (cols - 1) * gap
        toggle_w = max(self._si(120), available // cols if cols else available)
        for idx, toggle in enumerate(self.toggles):
            col = idx % cols
            row = idx // cols
            x0 = pad + col * (toggle_w + gap)
            y0 = y + row * (toggle_h + gap)
            x1 = x0 + toggle_w
            y1 = y0 + toggle_h
            self.toggle_boxes.append({"key": toggle.key, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
            if toggle.value:
                fill = (95, 155, 255)
                border = (175, 210, 255)
            else:
                fill = (50, 50, 55)
                border = (95, 95, 110)
            cv.rectangle(self.img, (x0, y0), (x1, y1), fill, -1)
            cv.rectangle(self.img, (x0, y0), (x1, y1), border, self._thickness(1))
            cv.putText(self.img, toggle.label, (x0 + self._si(12), y0 + self._si(28)), font,
                       0.52 * self.scale, text_color, self._thickness(1), cv.LINE_AA)
        rows = (len(self.toggles) + cols - 1) // cols
        return y + rows * (toggle_h + gap) - gap + self._si(6)

    def _draw_sliders(self, y, font, pad, text_color, sub_color):
        self.slider_boxes = []
        if not self.sliders:
            return y
        for slider in self.sliders:
            label_scale = 0.55 * self.scale
            label_thickness = self._thickness(1)
            label_y = y + self._si(18)
            cv.putText(self.img, slider.label, (pad, label_y), font, label_scale, sub_color,
                       label_thickness, cv.LINE_AA)
            value_text = slider.formatted()
            value_size = cv.getTextSize(value_text, font, label_scale, label_thickness)[0]
            value_x = self.w - pad - value_size[0]
            cv.putText(self.img, value_text, (value_x, label_y), font, label_scale, text_color,
                       label_thickness, cv.LINE_AA)
            track_x0 = pad
            track_x1 = self.w - pad - value_size[0] - self._si(24)
            track_x1 = max(track_x1, track_x0 + self._si(60))
            track_y = label_y + self._si(10)
            half_h = self._si(6)
            cv.rectangle(self.img, (track_x0, track_y - half_h), (track_x1, track_y + half_h),
                         (55, 55, 62), -1)
            cv.rectangle(self.img, (track_x0, track_y - half_h), (track_x1, track_y + half_h),
                         (105, 105, 115), self._thickness(1))
            handle_x = int(round(track_x0 + slider.normalized() * (track_x1 - track_x0)))
            cv.circle(self.img, (handle_x, track_y), self._si(8), (210, 210, 210), -1)
            cv.circle(self.img, (handle_x, track_y), self._si(8), (245, 245, 245), self._thickness(1))
            self.slider_boxes.append({"key": slider.key, "x0": track_x0, "x1": track_x1,
                                      "y0": track_y - self._si(14), "y1": track_y + self._si(14)})
            y = track_y + self._si(24)
        return y

    def _draw_buttons(self, y, font, pad, text_color, now):
        self.button_boxes = []
        if not self.buttons:
            return y
        gap = self._si(12)
        btn_h = self._si(46)
        cols = 2
        available = self.w - 2 * pad - (cols - 1) * gap
        btn_w = max(self._si(150), available // cols if cols else available)
        for idx, (label, action) in enumerate(self.buttons):
            col = idx % cols
            row = idx // cols
            x0 = pad + col * (btn_w + gap)
            y0 = y + row * (btn_h + gap)
            x1 = x0 + btn_w
            y1 = y0 + btn_h
            self.button_boxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                      "idx": idx, "action": action})
            base = (60, 60, 68)
            border = (120, 120, 130)
            highlight = idx == self.flash_idx and now < self.flash_until
            cv.rectangle(self.img, (x0, y0), (x1, y1), base, -1)
            cv.rectangle(self.img, (x0, y0), (x1, y1),
                         (90, 200, 150) if highlight else border,
                         self._thickness(2 if highlight else 1))
            cv.putText(self.img, label, (x0 + self._si(12), y0 + self._si(28)), font,
                       0.52 * self.scale, text_color, self._thickness(1), cv.LINE_AA)
        rows = (len(self.buttons) + cols - 1) // cols
        return y + rows * (btn_h + gap) - gap + self._si(6)

    def render(self):
        self.dirty = False
        if self.img.shape[0] != self.h or self.img.shape[1] != self.w:
            self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.img[:] = (19, 19, 24)
        pad = self._si(20)
        font = cv.FONT_HERSHEY_SIMPLEX
        accent = (155, 185, 255)
        text_color = (232, 232, 232)
        sub_color = (188, 188, 188)
        now = time.time()
        if self.flash_idx is not None and now >= self.flash_until:
            self.flash_idx = None
        y = pad + self._si(30)
        cv.putText(self.img, "HeadCC Control", (pad, y), font, 0.9 * self.scale, text_color,
                   self._thickness(2), cv.LINE_AA)
        y += self._si(8)
        cv.line(self.img, (pad, y), (self.w - pad, y), (65, 65, 75), self._thickness(1))
        y += self._si(18)
        y = self._draw_section_title(y, "Live status", font, accent, pad)
        for label, value in self.readouts:
            cv.putText(self.img, f"{label}: {value}", (pad, y), font, 0.55 * self.scale,
                       sub_color, self._thickness(1), cv.LINE_AA)
            y += self._si(22)
        y += self._si(6)
        y = self._draw_section_title(y, "MIDI output", font, accent, pad)
        cv.putText(self.img,
                   f"Centered CC 1/11/74   {self.centered[0]:3d}  {self.centered[1]:3d}  {self.centered[2]:3d}",
                   (pad, y), font, 0.53 * self.scale, sub_color, self._thickness(1), cv.LINE_AA)
        y += self._si(22)
        cv.putText(self.img,
                   f"Absolute CC 21/22/23  {self.abscc[0]:3d}  {self.abscc[1]:3d}  {self.abscc[2]:3d}",
                   (pad, y), font, 0.53 * self.scale, sub_color, self._thickness(1), cv.LINE_AA)
        y += self._si(28)
        y = self._draw_section_title(y, "Modes", font, accent, pad)
        y = self._draw_toggles(y, font, pad, text_color)
        y = self._draw_section_title(y, "Adjustments", font, accent, pad)
        y = self._draw_sliders(y, font, pad, text_color, sub_color)
        y = self._draw_section_title(y, "Quick actions", font, accent, pad)
        y = self._draw_buttons(y, font, pad, text_color, now)
        y = self._draw_section_title(y, "Activity", font, accent, pad)
        for line in self.status:
            cv.putText(self.img, line, (pad, y), font, 0.5 * self.scale, sub_color,
                       self._thickness(1), cv.LINE_AA)
            y += self._si(18)
        cv.imshow(self.win, self.img)


def draw_bars(frame, triplet, title, origin=(20, 80), scale=1.0):
    font = cv.FONT_HERSHEY_SIMPLEX
    x0 = int(round(origin[0] * scale))
    y0 = int(round(origin[1] * scale))
    w = int(round(340 * scale))
    h = max(1, int(round(22 * scale)))
    gap = max(1, int(round(12 * scale)))
    header_gap = int(round(34 * scale))
    overlay = frame.copy()
    top = max(0, y0 - header_gap)
    bottom = min(frame.shape[0] - 1, y0 + 3 * (h + gap) + int(round(18 * scale)))
    right = min(frame.shape[1] - 1, x0 + w + int(round(160 * scale)))
    cv.rectangle(overlay, (max(0, x0 - int(round(16 * scale))), top), (right, bottom),
                 (18, 18, 18), -1)
    cv.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv.putText(frame, title, (x0, y0 - int(round(14 * scale))), font,
               0.68 * scale, (235, 235, 235), max(1, int(round(2 * scale))), cv.LINE_AA)
    colors = [(255, 176, 108), (130, 220, 130), (120, 188, 255)]
    labels = ["Yaw", "Pitch", "Roll"]
    for i, (name, val, color) in enumerate(zip(labels, triplet, colors)):
        y = y0 + i * (h + gap)
        cv.rectangle(frame, (x0, y), (x0 + w, y + h), (90, 90, 90), max(1, int(round(2 * scale))))
        center_x = x0 + w // 2
        cv.line(frame, (center_x, y), (center_x, y + h), (115, 115, 115), max(1, int(round(scale))))
        fill = int(round((val / 127.0) * w))
        cv.rectangle(frame, (x0, y), (x0 + fill, y + h), color, -1)
        cv.putText(frame, f"{name} {val:3d}",
                   (x0 + w + int(round(14 * scale)), y + h - int(round(4 * scale))),
                   font, 0.55 * scale, (235, 235, 235), max(1, int(round(2 * scale))), cv.LINE_AA)


SLIDER_YAW = "yaw_span"
SLIDER_PITCH = "pitch_span"
SLIDER_ROLL = "roll_span"
SLIDER_GLOBAL = "global_sensitivity"
SLIDER_DEADZONE = "deadzone"
SLIDER_SMOOTH = "smoothing"
SLIDER_RATE = "send_rate"
SLIDER_CHANNEL = "midi_channel"

TOGGLE_SEND = "send_enabled"
TOGGLE_INV_YAW = "invert_yaw"
TOGGLE_INV_PITCH = "invert_pitch"
TOGGLE_INV_ROLL = "invert_roll"

BAR_BLOCK_BASE_OFFSET = 3 * (22 + 12) + 64


def apply_cfg_to_panel(panel, cfg):
    panel.set_slider_value(SLIDER_YAW, abs(cfg["yaw_range_deg"][1] - cfg["yaw_range_deg"][0]))
    panel.set_slider_value(SLIDER_PITCH, abs(cfg["pitch_range_deg"][1] - cfg["pitch_range_deg"][0]))
    panel.set_slider_value(SLIDER_ROLL, abs(cfg["roll_range_deg"][1] - cfg["roll_range_deg"][0]))
    panel.set_slider_value(SLIDER_GLOBAL, cfg.get("global_sens_pct", 100))
    panel.set_slider_value(SLIDER_DEADZONE, cfg.get("deadzone_deg", 0.0))
    panel.set_slider_value(SLIDER_SMOOTH, cfg.get("smooth_alpha", 0.0))
    panel.set_slider_value(SLIDER_RATE, cfg.get("send_rate_hz", 30))
    panel.set_slider_value(SLIDER_CHANNEL, cfg.get("midi_channel", 0) + 1)
    panel.set_toggle_state(TOGGLE_INV_YAW, cfg.get("invert_yaw", False))
    panel.set_toggle_state(TOGGLE_INV_PITCH, cfg.get("invert_pitch", False))
    panel.set_toggle_state(TOGGLE_INV_ROLL, cfg.get("invert_roll", False))
# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cam", type=int)
    ap.add_argument("--port", type=str)
    ap.add_argument("--showports", action="store_true")
    ap.add_argument("--ui-scale", type=float)
    args=ap.parse_args()

    cfg=load_cfg()
    if args.cam is not None: cfg["cam_index"]=args.cam
    if args.port: cfg["midi_port_substr"]=args.port
    if args.ui_scale is not None: cfg["ui_scale"] = args.ui_scale
    if args.showports: print(get_output_names()); return

    ui_scale = clamp(float(cfg.get("ui_scale", 1.0)), UI_SCALE_MIN, UI_SCALE_MAX)
    cfg["ui_scale"] = ui_scale

    port_name=find_midi_port(cfg["midi_port_substr"])
    midi=open_output(port_name)
    print(f"[MIDI] {port_name}")

    cap=open_camera(cfg["cam_index"], cfg["width"], cfg["height"], cfg["fps"])

    # Windows
    win="Head -> MIDI CC"; cv.namedWindow(win, cv.WINDOW_NORMAL); cv.moveWindow(win, 40, 40); cv.resizeWindow(win, int(round(1050*ui_scale)), int(round(700*ui_scale)))
    ctrl="Control Panel"; cv.namedWindow(ctrl, cv.WINDOW_NORMAL); cv.moveWindow(ctrl, 1120, 40)
    panel = ControlPanel(ctrl, scale=ui_scale)
    cv.resizeWindow(ctrl, panel.w, panel.h)

    def apply_ui_scale(new_scale):
        nonlocal ui_scale
        new_scale = clamp(float(new_scale), UI_SCALE_MIN, UI_SCALE_MAX)
        if abs(new_scale - ui_scale) < 1e-6:
            return False
        ui_scale = new_scale
        cfg["ui_scale"] = ui_scale
        cv.resizeWindow(win, int(round(1050*ui_scale)), int(round(700*ui_scale)))
        panel.set_scale(ui_scale)
        cv.resizeWindow(ctrl, panel.w, panel.h)
        panel.render_if_needed()
        return True


    def span_from_cfg(key, default):
        rng = cfg.get(key)
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                span = abs(float(rng[1]) - float(rng[0]))
                if span > 0:
                    return span
            except (TypeError, ValueError):
                pass
        return default

    slider_specs = [
        {"key": SLIDER_YAW, "label": "Yaw span", "min": 20.0, "max": 180.0, "step": 1.0,
         "value": span_from_cfg("yaw_range_deg", 90.0), "fmt": "{:.0f}°"},
        {"key": SLIDER_PITCH, "label": "Pitch span", "min": 20.0, "max": 180.0, "step": 1.0,
         "value": span_from_cfg("pitch_range_deg", 60.0), "fmt": "{:.0f}°"},
        {"key": SLIDER_ROLL, "label": "Roll span", "min": 20.0, "max": 180.0, "step": 1.0,
         "value": span_from_cfg("roll_range_deg", 70.0), "fmt": "{:.0f}°"},
        {"key": SLIDER_GLOBAL, "label": "Sensitivity", "min": 10.0, "max": 300.0, "step": 1.0,
         "value": cfg.get("global_sens_pct", 100), "fmt": "{:.0f}%", "cast": int},
        {"key": SLIDER_DEADZONE, "label": "Deadzone", "min": 0.0, "max": 10.0, "step": 0.1,
         "value": cfg.get("deadzone_deg", 2.0), "fmt": "{:.1f}°"},
        {"key": SLIDER_SMOOTH, "label": "Smoothing", "min": 0.0, "max": 1.0, "step": 0.01,
         "value": cfg.get("smooth_alpha", 0.25), "fmt": "{:.2f}"},
        {"key": SLIDER_RATE, "label": "Send rate", "min": 1.0, "max": 120.0, "step": 1.0,
         "value": cfg.get("send_rate_hz", 30), "fmt": "{:.0f} Hz", "cast": int},
        {"key": SLIDER_CHANNEL, "label": "MIDI channel", "min": 1.0, "max": 16.0, "step": 1.0,
         "value": cfg.get("midi_channel", 0) + 1, "fmt": "Ch {:02.0f}", "cast": int},
    ]
    panel.set_sliders(slider_specs)
    panel.set_toggles([
        {"key": TOGGLE_SEND, "label": "Send MIDI", "value": True},
        {"key": TOGGLE_INV_YAW, "label": "Invert yaw", "value": cfg.get("invert_yaw", False)},
        {"key": TOGGLE_INV_PITCH, "label": "Invert pitch", "value": cfg.get("invert_pitch", False)},
        {"key": TOGGLE_INV_ROLL, "label": "Invert roll", "value": cfg.get("invert_roll", False)},
    ])
    panel.set_buttons([
        ("Calibrate origin", "calibrate"),
        ("Save settings", "save"),
        ("Reset defaults", "reset"),
        ("Burst current CCs", "burst_cc"),
        ("Send CC1 (Yaw)", f"cc:{cfg['cc_yaw']}"),
        ("Send CC11 (Pitch)", f"cc:{cfg['cc_pitch']}"),
        ("Send CC74 (Roll)", f"cc:{cfg['cc_roll']}"),
        ("Send CC21 (YawAbs)", f"cc:{cfg['cc_yaw_abs']}"),
        ("Send CC22 (PitchAbs)", f"cc:{cfg['cc_pitch_abs']}"),
        ("Send CC23 (RollAbs)", f"cc:{cfg['cc_roll_abs']}"),
    ])
    apply_cfg_to_panel(panel, cfg)
    panel.set_toggle_state(TOGGLE_SEND, True)
    panel.notify("Controls ready. Space toggles send or use the panel toggle.")
    panel.notify("Press C or the Calibrate button to set a neutral pose.")
    panel.notify("Y/P/R flip axes; +/- adjust UI scale; Q quits.")
    panel.render_if_needed()

    send_enabled_prev = panel.get_toggle_state(TOGGLE_SEND)
    invert_prev = {
        TOGGLE_INV_YAW: panel.get_toggle_state(TOGGLE_INV_YAW),
        TOGGLE_INV_PITCH: panel.get_toggle_state(TOGGLE_INV_PITCH),
        TOGGLE_INV_ROLL: panel.get_toggle_state(TOGGLE_INV_ROLL),
    }

    # FaceMesh
    mp_face = mp_solutions.face_mesh
    face = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp_solutions.drawing_utils
    mp_styles = mp_solutions.drawing_styles
    mesh_spec = mp_styles.get_default_face_mesh_tesselation_style()
    contour_spec = mp_styles.get_default_face_mesh_contours_style()
    iris_spec = mp_styles.get_default_face_mesh_iris_connections_style()

    # Pose state
    prev_rvec = prev_tvec = None
    R0 = None           # baseline rotation
    eul_prev = None     # unwrap continuity
    yaw_s=pitch_s=roll_s=None
    last_send=0.0
    centered=(64,64,64); abscc=(0,0,0)

    def send_cc_values(center_vals, abs_vals):
        ch = cfg["midi_channel"]
        for cc, val in zip((cfg["cc_yaw"], cfg["cc_pitch"], cfg["cc_roll"]), center_vals):
            midi.send(Message('control_change', channel=ch, control=cc, value=int(val)))
        for cc, val in zip((cfg["cc_yaw_abs"], cfg["cc_pitch_abs"], cfg["cc_roll_abs"]), abs_vals):
            midi.send(Message('control_change', channel=ch, control=cc, value=int(val)))
        return ch

    try:
        while True:
            ok, frame = cap.read()
            if not ok: print("[ERR] frame read"); break
            h,w = frame.shape[:2]
            res = face.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))


            # read controls from panel
            def slider_val(key, default):
                value = panel.get_slider_value(key)
                return default if value is None else value

            base_yaw_span = float(slider_val(SLIDER_YAW, span_from_cfg("yaw_range_deg", 90.0)))
            base_pitch_span = float(slider_val(SLIDER_PITCH, span_from_cfg("pitch_range_deg", 60.0)))
            base_roll_span = float(slider_val(SLIDER_ROLL, span_from_cfg("roll_range_deg", 70.0)))
            cfg["global_sens_pct"] = int(slider_val(SLIDER_GLOBAL, cfg.get("global_sens_pct", 100)))
            cfg["deadzone_deg"] = float(slider_val(SLIDER_DEADZONE, cfg.get("deadzone_deg", 2.0)))
            cfg["smooth_alpha"] = float(slider_val(SLIDER_SMOOTH, cfg.get("smooth_alpha", 0.25)))
            cfg["send_rate_hz"] = max(1, int(slider_val(SLIDER_RATE, cfg.get("send_rate_hz", 30))))
            channel_val = int(slider_val(SLIDER_CHANNEL, cfg.get("midi_channel", 0) + 1))
            cfg["midi_channel"] = clamp(channel_val - 1, 0, 15)

            send_enabled = panel.get_toggle_state(TOGGLE_SEND)
            if send_enabled != send_enabled_prev:
                panel.notify("Send enabled" if send_enabled else "Send muted")
                send_enabled_prev = send_enabled
            send_state = "On" if send_enabled else "Muted"

            invert_states = {
                TOGGLE_INV_YAW: panel.get_toggle_state(TOGGLE_INV_YAW),
                TOGGLE_INV_PITCH: panel.get_toggle_state(TOGGLE_INV_PITCH),
                TOGGLE_INV_ROLL: panel.get_toggle_state(TOGGLE_INV_ROLL),
            }
            for key, label in [
                (TOGGLE_INV_YAW, "Invert yaw"),
                (TOGGLE_INV_PITCH, "Invert pitch"),
                (TOGGLE_INV_ROLL, "Invert roll"),
            ]:
                if invert_states[key] != invert_prev[key]:
                    panel.notify(f"{label}: {invert_states[key]}")
                    invert_prev[key] = invert_states[key]
            cfg["invert_yaw"] = invert_states[TOGGLE_INV_YAW]
            cfg["invert_pitch"] = invert_states[TOGGLE_INV_PITCH]
            cfg["invert_roll"] = invert_states[TOGGLE_INV_ROLL]

            # keep base spans in cfg (so save/restore matches sliders)
            cfg["yaw_range_deg"] = [-base_yaw_span/2.0, base_yaw_span/2.0]
            cfg["pitch_range_deg"] = [-base_pitch_span/2.0, base_pitch_span/2.0]
            cfg["roll_range_deg"] = [-base_roll_span/2.0, base_roll_span/2.0]

            # effective spans after global sensitivity
            yaw_rng   = eff_range(cfg["yaw_range_deg"],   cfg["global_sens_pct"])
            pitch_rng = eff_range(cfg["pitch_range_deg"], cfg["global_sens_pct"])
            roll_rng  = eff_range(cfg["roll_range_deg"],  cfg["global_sens_pct"])

            pose_ok=False
            face_landmarks=None
            if res.multi_face_landmarks:
                face_landmarks = res.multi_face_landmarks[0]
                lms = face_landmarks.landmark
                pts2d = np.array([(lms[i].x*w, lms[i].y*h) for i in LM], dtype=np.float32)
                R, prev_rvec, prev_tvec = solve_pose(pts2d, w, h, cfg["fov_deg"], prev_rvec, prev_tvec)
                if R is not None:
                    if R0 is None: R0 = R.copy()  # auto-baseline first time
                    R_rel = R0.T @ R
                    yaw, pitch, roll = euler_from_R_rel(R_rel)
                    # fuse roll with eye-line for stability
                    roll = 0.7*roll + 0.3*roll_from_eyes_2d(lms, w, h)
                    # unwrap and invert
                    yaw, pitch, roll = unwrap(eul_prev, (yaw, pitch, roll)); eul_prev = (yaw, pitch, roll)
                    if cfg["invert_yaw"]:   yaw = -yaw
                    if cfg["invert_pitch"]: pitch = -pitch
                    if cfg["invert_roll"]:  roll = -roll
                    # smooth
                    yaw_s   = ema(yaw_s,   yaw,   cfg["smooth_alpha"])
                    pitch_s = ema(pitch_s, pitch, cfg["smooth_alpha"])
                    roll_s  = ema(roll_s,  roll,  cfg["smooth_alpha"])
                    pose_ok=True

            # HUD
            if face_landmarks is not None:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mesh_spec,
                )
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=contour_spec,
                )
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=iris_spec,
                )

            if pose_ok:
                cv.putText(frame, f"Yaw {yaw_s:6.1f}  Pitch {pitch_s:6.1f}  Roll {roll_s:6.1f}",
                           (int(round(20*ui_scale)), int(round(40*ui_scale))), cv.FONT_HERSHEY_SIMPLEX,
                           1.0 * ui_scale, (0,255,0), max(1, int(round(2*ui_scale))), cv.LINE_AA)
                for x,y in pts2d.astype(int): cv.circle(frame,(x,y),3,(0,255,0),-1)
            else:
                cv.putText(frame, "No face", (int(round(20*ui_scale)), int(round(40*ui_scale))),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0 * ui_scale, (0,0,255),
                           max(1, int(round(2*ui_scale))), cv.LINE_AA)

            # CC mapping
            now = time.time()
            if pose_ok:
                def dz(v):
                    return 0.0 if abs(v) < cfg["deadzone_deg"] else v

                yv, pv, rv = dz(yaw_s), dz(pitch_s), dz(roll_s)
                cy = map_centered(yv, yaw_rng)
                cp = map_centered(pv, pitch_rng)
                cr = map_centered(rv, roll_rng)
                ay = map_abs(yv, yaw_rng)
                ap = map_abs(pv, pitch_rng)
                ar = map_abs(rv, roll_rng)
                centered = (cy, cp, cr)
                abscc = (ay, ap, ar)
            else:
                centered = (64, 64, 64)
                abscc = (0, 0, 0)

            if pose_ok and send_enabled and (now - last_send) >= 1.0 / cfg["send_rate_hz"]:
                last_send = now
                send_cc_values(centered, abscc)

            panel.update_cc_values(centered, abscc)

            # bars
            draw_bars(frame, centered, "Centered CC · 1/11/74", origin=(20,80), scale=ui_scale)
            draw_bars(frame, abscc,   "Absolute CC · 21/22/23", origin=(20, 80 + BAR_BLOCK_BASE_OFFSET), scale=ui_scale)

            footer = (f"Send {send_state} • Ch {cfg['midi_channel']+1} • {cfg['send_rate_hz']}Hz • "
                      f"dz {cfg['deadzone_deg']:.1f}° • smooth {cfg['smooth_alpha']:.2f} • sens {cfg['global_sens_pct']}% • "
                      f"UI {ui_scale:.2f}x")
            cv.putText(frame, footer, (int(round(20*ui_scale)), h - int(round(20*ui_scale))),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6 * ui_scale, (200,200,200),
                       max(1, int(round(1*ui_scale))), cv.LINE_AA)
            cv.imshow(win, frame)


            # Button actions
            act = panel.consume_action()
            if act:
                ch = cfg["midi_channel"]
                if act == "burst_cc":
                    ch = send_cc_values(centered, abscc)
                    last_send = time.time()
                    panel.notify(f"Manual CC burst sent (ch {ch+1})")
                elif act.startswith("cc:"):
                    cc = int(act.split(":")[1])
                    midi.send(Message('control_change', channel=ch, control=cc, value=127))
                    midi.send(Message('control_change', channel=ch, control=cc, value=0))
                    panel.notify(f"Learn pulse sent on CC{cc} (ch {ch+1})")
                elif act == "save":
                    save_cfg(cfg)
                    panel.notify("Settings saved")
                elif act == "reset":
                    keep = {"midi_port_substr": cfg["midi_port_substr"], "cam_index": cfg["cam_index"]}
                    cfg.clear()
                    cfg.update(DEFAULTS)
                    cfg.update(keep)
                    apply_cfg_to_panel(panel, cfg)
                    panel.set_toggle_state(TOGGLE_SEND, True)
                    send_enabled_prev = panel.get_toggle_state(TOGGLE_SEND)
                    send_enabled = send_enabled_prev
                    invert_prev[TOGGLE_INV_YAW] = panel.get_toggle_state(TOGGLE_INV_YAW)
                    invert_prev[TOGGLE_INV_PITCH] = panel.get_toggle_state(TOGGLE_INV_PITCH)
                    invert_prev[TOGGLE_INV_ROLL] = panel.get_toggle_state(TOGGLE_INV_ROLL)
                    R0 = None
                    eul_prev = None
                    yaw_s = pitch_s = roll_s = 0.0
                    save_cfg(cfg)
                    panel.notify("Settings reset to defaults")
                elif act == "calibrate":
                    if pose_ok:
                        R0 = R.copy()
                        eul_prev = (0.0, 0.0, 0.0)
                        yaw_s = pitch_s = roll_s = 0.0
                        panel.notify("Origin calibrated")
                    else:
                        panel.notify("Need a face to calibrate origin")

            # keys
            k=cv.waitKey(1)&0xFF
            if k in (ord('q'),27): break
            elif k==ord('c') and pose_ok:
                R0 = R.copy()
                eul_prev = (0.0,0.0,0.0)
                yaw_s = pitch_s = roll_s = 0.0
                panel.notify("Origin calibrated")
            elif k==ord(' '):
                new_state = not panel.get_toggle_state(TOGGLE_SEND)
                panel.set_toggle_state(TOGGLE_SEND, new_state)
                send_enabled = new_state
                send_enabled_prev = new_state
                panel.notify("Send enabled" if new_state else "Send muted")
            elif k==ord('y'):
                new_state = not panel.get_toggle_state(TOGGLE_INV_YAW)
                panel.set_toggle_state(TOGGLE_INV_YAW, new_state)
                invert_prev[TOGGLE_INV_YAW] = new_state
                panel.notify(f"Invert yaw: {new_state}")
            elif k==ord('p'):
                new_state = not panel.get_toggle_state(TOGGLE_INV_PITCH)
                panel.set_toggle_state(TOGGLE_INV_PITCH, new_state)
                invert_prev[TOGGLE_INV_PITCH] = new_state
                panel.notify(f"Invert pitch: {new_state}")
            elif k==ord('r'):
                new_state = not panel.get_toggle_state(TOGGLE_INV_ROLL)
                panel.set_toggle_state(TOGGLE_INV_ROLL, new_state)
                invert_prev[TOGGLE_INV_ROLL] = new_state
                panel.notify(f"Invert roll: {new_state}")
            elif k==ord('s'):
                save_cfg(cfg); panel.notify("Settings saved")
            elif k in (ord('['), ord('{'), ord('-'), ord('_')):
                if apply_ui_scale(ui_scale - 0.1):
                    panel.notify(f"UI scale: {ui_scale:.2f}x")
            elif k in (ord(']'), ord('}'), ord('+'), ord('=')):
                if apply_ui_scale(ui_scale + 0.1):
                    panel.notify(f"UI scale: {ui_scale:.2f}x")

            status = "Tracking" if pose_ok else "No face"
            yaw_half = max(abs(yaw_rng[0]), abs(yaw_rng[1]))
            pitch_half = max(abs(pitch_rng[0]), abs(pitch_rng[1]))
            roll_half = max(abs(roll_rng[0]), abs(roll_rng[1]))
            panel.update_readouts([
                ("Face", status),
                ("Send", send_state),
                ("Channel", f"{cfg['midi_channel']+1}"),
                ("Rate", f"{cfg['send_rate_hz']} Hz"),
                ("Yaw range", f"±{yaw_half:.0f}°"),
                ("Pitch range", f"±{pitch_half:.0f}°"),
                ("Roll range", f"±{roll_half:.0f}°"),
                ("Sensitivity", f"{cfg['global_sens_pct']}%"),
                ("Deadzone", f"{cfg['deadzone_deg']:.1f}°"),
                ("Smooth", f"{cfg['smooth_alpha']:.2f}"),
                ("UI scale", f"{ui_scale:.2f}x"),
            ])
            panel.render_if_needed()

    finally:
        # auto-save on exit
        save_cfg(cfg)
        cap.release(); cv.destroyAllWindows(); midi.close()

if __name__ == "__main__":
    main()
