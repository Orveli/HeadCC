# HeadCC.py — robust baseline, 6x CC, mute toggle, global sensitivity, control panel, save/reset
# Python 3.10 x64, mediapipe 0.10.x
import os, json, time, argparse, collections
import numpy as np
import cv2 as cv
import mediapipe as mp
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
    "cam_index": 0, "width": 1280, "height": 720, "fps": 60, "fov_deg": 60.0
}

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
class ControlPanel:
    def __init__(self, window, w=420, h=620, status_lines=5):
        self.win = window
        self.w, self.h = w, h
        self.img = np.zeros((h, w, 3), dtype=np.uint8)
        self.buttons = []  # list of (label, action)
        self.button_boxes = []  # cached layout (x0,y0,x1,y1,label,action)
        self.clicked_action = None
        self.readouts = []
        self.centered = (0, 0, 0)
        self.abscc = (0, 0, 0)
        self.status = collections.deque(maxlen=status_lines)
        self.dirty = True
        self.flash_idx = None
        self.flash_until = 0.0
        cv.setMouseCallback(self.win, self.on_mouse)

    def set_buttons(self, items):
        self.buttons = list(items)
        self.button_boxes = []
        self.mark_dirty()

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

    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            for idx, (x0, y0, x1, y1, label, action) in enumerate(self.button_boxes):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.clicked_action = action
                    self.flash_idx = idx
                    self.flash_until = time.time() + 0.2
                    self.mark_dirty()
                    break

    def render_if_needed(self):
        if self.dirty:
            self.render()

    def render(self):
        self.dirty = False
        self.img[:] = (28, 28, 28)
        font = cv.FONT_HERSHEY_SIMPLEX

        cv.putText(self.img, "Control Panel", (20, 34), font, 0.8, (250, 250, 250), 2, cv.LINE_AA)

        y = 64
        for label, value in self.readouts:
            cv.putText(self.img, f"{label}: {value}", (20, y), font, 0.55, (220, 220, 220), 1, cv.LINE_AA)
            y += 24

        y += 6
        cv.putText(self.img, "CC output", (20, y), font, 0.6, (200, 210, 255), 1, cv.LINE_AA)
        y += 24
        cv.putText(self.img,
                   f"Centered CC1/11/74  {self.centered[0]:3d}  {self.centered[1]:3d}  {self.centered[2]:3d}",
                   (20, y), font, 0.5, (210, 210, 210), 1, cv.LINE_AA)
        y += 20
        cv.putText(self.img,
                   f"Absolute CC21/22/23 {self.abscc[0]:3d}  {self.abscc[1]:3d}  {self.abscc[2]:3d}",
                   (20, y), font, 0.5, (210, 210, 210), 1, cv.LINE_AA)

        y += 32
        cv.putText(self.img, "Actions", (20, y), font, 0.6, (200, 210, 255), 1, cv.LINE_AA)
        y += 10

        now = time.time()
        btn_width = self.w - 40
        btn_height = 36
        gap = 10
        x0 = 20
        start_y = y + 4
        self.button_boxes = []
        for idx, (label, action) in enumerate(self.buttons):
            y0 = start_y + idx * (btn_height + gap)
            y1 = y0 + btn_height
            x1 = x0 + btn_width
            self.button_boxes.append((x0, y0, x1, y1, label, action))
            cv.rectangle(self.img, (x0, y0), (x1, y1), (200, 200, 200), 1)
            if idx == self.flash_idx and now < self.flash_until:
                cv.rectangle(self.img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv.putText(self.img, label, (x0 + 10, y0 + 24), font, 0.55, (230, 230, 230), 1, cv.LINE_AA)

        y = start_y + len(self.buttons) * (btn_height + gap) + 16
        if y > self.h - 90:
            y = self.h - 90
        cv.putText(self.img, "Status", (20, y), font, 0.6, (200, 210, 255), 1, cv.LINE_AA)
        y += 24
        for line in self.status:
            cv.putText(self.img, line, (20, y), font, 0.5, (210, 210, 210), 1, cv.LINE_AA)
            y += 18

        cv.imshow(self.win, self.img)

def draw_bars(frame, triplet, title, origin=(20,80)):
    x0,y0=origin; w=360; h=20; gap=10; font=cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, title, (x0, y0-12), font, 0.6, (200,200,200), 1, cv.LINE_AA)
    labels=[("Yaw",triplet[0]),("Pitch",triplet[1]),("Roll",triplet[2])]
    for i,(name,val) in enumerate(labels):
        y=y0+i*(h+gap)
        cv.rectangle(frame,(x0,y),(x0+w,y+h),(160,160,160),1)
        fill=int((val/127.0)*w)
        cv.rectangle(frame,(x0,y),(x0+fill,y+h),(0,255,0),-1)
        cv.putText(frame,f"{name} {val:3d}",(x0+w+10,y+h-4),font,0.55,(255,255,255),1,cv.LINE_AA)

TB_YAW = "Yaw span (deg)"
TB_PITCH = "Pitch span (deg)"
TB_ROLL = "Roll span (deg)"
TB_GLOBAL = "Sensitivity (%)"
TB_DEADZONE = "Deadzone (0.1deg)"
TB_SMOOTH = "Smoothing (x100)"
TB_RATE = "Send rate (Hz)"
TB_SEND = "Send on/off"
TB_CHAN = "MIDI channel"

def set_trackbars_from_cfg(ctrl, cfg):
    def total_span(r): return int(round(abs(r[1]-r[0])))
    cv.setTrackbarPos(TB_YAW,   ctrl, total_span(cfg["yaw_range_deg"]))
    cv.setTrackbarPos(TB_PITCH, ctrl, total_span(cfg["pitch_range_deg"]))
    cv.setTrackbarPos(TB_ROLL,  ctrl, total_span(cfg["roll_range_deg"]))
    cv.setTrackbarPos(TB_DEADZONE, ctrl, int(cfg["deadzone_deg"]*10))
    cv.setTrackbarPos(TB_SMOOTH,   ctrl, int(cfg["smooth_alpha"]*100))
    cv.setTrackbarPos(TB_RATE,       ctrl, int(cfg["send_rate_hz"]))
    cv.setTrackbarPos(TB_GLOBAL, ctrl, int(cfg["global_sens_pct"]))
    cv.setTrackbarPos(TB_CHAN, ctrl, int(cfg["midi_channel"]+1))

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cam", type=int)
    ap.add_argument("--port", type=str)
    ap.add_argument("--showports", action="store_true")
    args=ap.parse_args()

    cfg=load_cfg()
    if args.cam is not None: cfg["cam_index"]=args.cam
    if args.port: cfg["midi_port_substr"]=args.port
    if args.showports: print(get_output_names()); return

    port_name=find_midi_port(cfg["midi_port_substr"])
    midi=open_output(port_name)
    print(f"[MIDI] {port_name}")

    cap=open_camera(cfg["cam_index"], cfg["width"], cfg["height"], cfg["fps"])

    # Windows
    win="Head -> MIDI CC"; cv.namedWindow(win, cv.WINDOW_NORMAL); cv.moveWindow(win, 40, 40); cv.resizeWindow(win, 1050, 700)
    ctrl="Control Panel"; cv.namedWindow(ctrl, cv.WINDOW_NORMAL); cv.moveWindow(ctrl, 1120, 40); cv.resizeWindow(ctrl, 420, 640)
    panel = ControlPanel(ctrl, 420, 640)

    # Trackbars
    def mk_tb(name, init, maxv): cv.createTrackbar(name, ctrl, init, maxv, lambda *_: None)
    # keep labels short so they fit
    def init_span_tb(label, key, default_span):
        v = int(round(abs(cfg[key][1]-cfg[key][0]))) or default_span
        v = int(clamp(v, 10, 180))
        mk_tb(label, v, 180)

    init_span_tb(TB_YAW,   "yaw_range_deg",   90)
    init_span_tb(TB_PITCH, "pitch_range_deg", 60)
    init_span_tb(TB_ROLL,  "roll_range_deg",  70)
    mk_tb(TB_GLOBAL, int(cfg["global_sens_pct"]), 300)
    mk_tb(TB_DEADZONE, int(cfg["deadzone_deg"]*10), 100)
    mk_tb(TB_SMOOTH,   int(cfg["smooth_alpha"]*100), 100)
    mk_tb(TB_RATE,       int(cfg["send_rate_hz"]), 120)
    mk_tb(TB_SEND,      1, 1)  # mute toggle (space toggles too)
    mk_tb(TB_CHAN, int(cfg["midi_channel"]+1), 16)

    # Tools buttons
    rows = [
        ("Send CC1 (Yaw)",          f"cc:{cfg['cc_yaw']}"),
        ("Send CC11 (Pitch)",       f"cc:{cfg['cc_pitch']}"),
        ("Send CC74 (Roll)",        f"cc:{cfg['cc_roll']}"),
        ("Send CC21 (YawAbs)",      f"cc:{cfg['cc_yaw_abs']}"),
        ("Send CC22 (PitchAbs)",    f"cc:{cfg['cc_pitch_abs']}"),
        ("Send CC23 (RollAbs)",     f"cc:{cfg['cc_roll_abs']}"),
        ("Save settings",           "save"),
        ("Reset settings",          "reset"),
    ]
    panel.set_buttons(rows)
    panel.notify("Controls ready. Space toggles send, Q quits.")
    panel.notify("Press C to recalibrate origin; Y/P/R invert axes.")
    panel.render_if_needed()

    # FaceMesh
    mp_face=mp.solutions.face_mesh
    face=mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                          min_detection_confidence=0.6, min_tracking_confidence=0.6)

    # Pose state
    prev_rvec = prev_tvec = None
    R0 = None           # baseline rotation
    eul_prev = None     # unwrap continuity
    yaw_s=pitch_s=roll_s=None
    last_send=0.0
    centered=(64,64,64); abscc=(0,0,0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: print("[ERR] frame read"); break
            h,w = frame.shape[:2]
            res = face.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            # read controls
            def span_from_tb(tb):
                return float(clamp(cv.getTrackbarPos(tb, ctrl), 10, 180))
            base_yaw_span   = span_from_tb(TB_YAW)
            base_pitch_span = span_from_tb(TB_PITCH)
            base_roll_span  = span_from_tb(TB_ROLL)
            cfg["global_sens_pct"] = clamp(cv.getTrackbarPos(TB_GLOBAL, ctrl), 10, 300)
            cfg["deadzone_deg"]    = cv.getTrackbarPos(TB_DEADZONE, ctrl)/10.0
            cfg["smooth_alpha"]    = cv.getTrackbarPos(TB_SMOOTH,   ctrl)/100.0
            cfg["send_rate_hz"]    = max(1, cv.getTrackbarPos(TB_RATE, ctrl) or 30)
            send_enabled           = cv.getTrackbarPos(TB_SEND, ctrl) == 1
            cfg["midi_channel"]    = clamp(cv.getTrackbarPos(TB_CHAN, ctrl)-1, 0, 15)

            # keep base spans in cfg (so save/restore matches sliders)
            cfg["yaw_range_deg"]   = [-base_yaw_span/2.0,   base_yaw_span/2.0]
            cfg["pitch_range_deg"] = [-base_pitch_span/2.0, base_pitch_span/2.0]
            cfg["roll_range_deg"]  = [-base_roll_span/2.0,  base_roll_span/2.0]

            # effective spans after global sensitivity
            yaw_rng   = eff_range(cfg["yaw_range_deg"],   cfg["global_sens_pct"])
            pitch_rng = eff_range(cfg["pitch_range_deg"], cfg["global_sens_pct"])
            roll_rng  = eff_range(cfg["roll_range_deg"],  cfg["global_sens_pct"])

            pose_ok=False
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
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
            if pose_ok:
                cv.putText(frame, f"Yaw {yaw_s:6.1f}  Pitch {pitch_s:6.1f}  Roll {roll_s:6.1f}",
                           (20,40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv.LINE_AA)
                for x,y in pts2d.astype(int): cv.circle(frame,(x,y),3,(0,255,0),-1)
            else:
                cv.putText(frame, "No face", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv.LINE_AA)

            # CC mapping
            now = time.time()
            if pose_ok and send_enabled and (now - last_send) >= 1.0/cfg["send_rate_hz"]:
                last_send=now
                def dz(v): return 0.0 if abs(v) < cfg["deadzone_deg"] else v
                yv, pv, rv = dz(yaw_s), dz(pitch_s), dz(roll_s)

                cy = map_centered(yv, yaw_rng)
                cp = map_centered(pv, pitch_rng)
                cr = map_centered(rv, roll_rng)
                centered=(cy,cp,cr)

                ay = map_abs(yv, yaw_rng)
                ap = map_abs(pv, pitch_rng)
                ar = map_abs(rv, roll_rng)
                abscc=(ay,ap,ar)

                ch=cfg["midi_channel"]
                # centered
                for cc,val,label in [(cfg["cc_yaw"],cy,"Yaw"),
                                     (cfg["cc_pitch"],cp,"Pitch"),
                                     (cfg["cc_roll"],cr,"Roll")]:
                    midi.send(Message('control_change', channel=ch, control=cc, value=val))
                # absolute
                for cc,val,label in [(cfg["cc_yaw_abs"],ay,"YawAbs"),
                                     (cfg["cc_pitch_abs"],ap,"PitchAbs"),
                                     (cfg["cc_roll_abs"],ar,"RollAbs")]:
                    midi.send(Message('control_change', channel=ch, control=cc, value=val))

            panel.update_cc_values(centered, abscc)

            # bars
            draw_bars(frame, centered, "Centered CC  CC1/11/74", origin=(20,80))
            draw_bars(frame, abscc,   "ABS CC       CC21/22/23", origin=(20, 80+3*(20+10)+26))

            footer = f"Send={int(send_enabled)} | Ch={cfg['midi_channel']+1} | rate {cfg['send_rate_hz']}Hz | dz {cfg['deadzone_deg']:.1f}° | smooth {cfg['smooth_alpha']:.2f} | sens {cfg['global_sens_pct']}%"
            cv.putText(frame, footer, (20, h-20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv.LINE_AA)
            cv.imshow(win, frame)

            # Button actions
            act = panel.consume_action()
            if act:
                ch = cfg["midi_channel"]
                if act.startswith("cc:"):
                    cc = int(act.split(":")[1])
                    # emit a short pair for MIDI-learn
                    midi.send(Message('control_change', channel=ch, control=cc, value=127))
                    midi.send(Message('control_change', channel=ch, control=cc, value=0))
                    panel.notify(f"Learn pulse sent on CC{cc} (ch {ch+1})")
                elif act == "save":
                    save_cfg(cfg); panel.notify("Settings saved")
                elif act == "reset":
                    # reset to defaults, keep port/cam
                    keep = {"midi_port_substr": cfg["midi_port_substr"], "cam_index": cfg["cam_index"]}
                    cfg.clear(); cfg.update(DEFAULTS); cfg.update(keep)
                    set_trackbars_from_cfg(ctrl, cfg)
                    # clear baseline and filters
                    R0=None; eul_prev=None; yaw_s=pitch_s=roll_s=0.0
                    save_cfg(cfg); panel.notify("Settings reset to defaults")
                    panel.mark_dirty()

            # keys
            k=cv.waitKey(1)&0xFF
            if k in (ord('q'),27): break
            elif k==ord('c') and pose_ok:
                R0 = R.copy()
                eul_prev = (0.0,0.0,0.0)
                yaw_s = pitch_s = roll_s = 0.0
                panel.notify("Origin calibrated")
            elif k==ord(' '):
                cv.setTrackbarPos(TB_SEND, ctrl, 0 if send_enabled else 1)
                panel.notify("Send muted" if send_enabled else "Send enabled")
            elif k==ord('y'):
                cfg["invert_yaw"]=not cfg["invert_yaw"];   panel.notify(f"Invert yaw: {cfg['invert_yaw']}")
            elif k==ord('p'):
                cfg["invert_pitch"]=not cfg["invert_pitch"]; panel.notify(f"Invert pitch: {cfg['invert_pitch']}")
            elif k==ord('r'):
                cfg["invert_roll"]=not cfg["invert_roll"];   panel.notify(f"Invert roll: {cfg['invert_roll']}")
            elif k==ord('s'):
                save_cfg(cfg); panel.notify("Settings saved")

            status = "Tracking" if pose_ok else "No face"
            send_state = "On" if send_enabled else "Muted"
            yaw_half = max(abs(yaw_rng[0]), abs(yaw_rng[1]))
            pitch_half = max(abs(pitch_rng[0]), abs(pitch_rng[1]))
            roll_half = max(abs(roll_rng[0]), abs(roll_rng[1]))
            panel.update_readouts([
                ("Yaw range", f"±{yaw_half:.0f}°"),
                ("Pitch range", f"±{pitch_half:.0f}°"),
                ("Roll range", f"±{roll_half:.0f}°"),
                ("Sensitivity", f"{cfg['global_sens_pct']}%"),
                ("Deadzone", f"{cfg['deadzone_deg']:.1f}°"),
                ("Smooth", f"{cfg['smooth_alpha']:.2f}"),
                ("Rate", f"{cfg['send_rate_hz']} Hz"),
                ("Channel", f"{cfg['midi_channel']+1}"),
                ("Send", send_state),
                ("Face", status),
            ])
            panel.render_if_needed()

    finally:
        # auto-save on exit
        save_cfg(cfg)
        cap.release(); cv.destroyAllWindows(); midi.close()

if __name__ == "__main__":
    main()
