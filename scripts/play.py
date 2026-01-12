import os
import sys
import time
import json
import threading
import ctypes
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO
from nitrogen.inference_viz import create_viz, VideoRecorder
from nitrogen.inference_client import ModelClient

import argparse
parser = argparse.ArgumentParser(description="VLM Inference")
# parser.add_argument("--process", type=str, default="celeste.exe", help="Game to play")
# parser.add_argument("--process", type=str, default="isaac-ng.exe", help="Game to play")
parser.add_argument("--process", type=str, default="Cuphead.exe", help="Game to play")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu actions (Disabled by default)")
parser.add_argument("--port", type=int, default=5555, help="Port for model server")
parser.add_argument("--manual", default=True, action="store_true", help="Start in manual controller-forwarding mode")
parser.add_argument("--manual-hotkey", type=str, default="F8", help="Hotkey to toggle manual/model mode (default: F8)")

args = parser.parse_args()

def _vk_code_from_name(name: str) -> int | None:
    n = name.strip().upper()
    if n.startswith("F") and n[1:].isdigit():
        fnum = int(n[1:])
        if 1 <= fnum <= 24:
            return 0x70 + (fnum - 1)  # VK_F1..VK_F24
    if n in {"ESC", "ESCAPE"}:
        return 0x1B
    if n in {"TAB"}:
        return 0x09
    if n in {"SPACE"}:
        return 0x20
    return None


class PhysicalGamepadForwarder:
    """
    Read a physical gamepad and convert its current state into NitroGen's unified action dict.
    This is meant to be used with GamepadEnv, which ultimately writes to the virtual vgamepad controller.
    """

    def __init__(self, controller_index: int = 0):
        try:
            from inputs import devices
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'inputs'. Install with: python -m pip install inputs"
            ) from e

        gamepads = list(getattr(devices, "gamepads", []))
        if not gamepads:
            raise RuntimeError("No physical gamepads detected by 'inputs'.")
        if controller_index < 0 or controller_index >= len(gamepads):
            raise RuntimeError(
                f"manual-controller-index out of range. Found {len(gamepads)} gamepad(s): "
                f"{[getattr(g, 'name', 'unknown') for g in gamepads]}"
            )

        self.device = gamepads[controller_index]
        self.device_name = getattr(self.device, "name", "unknown")

        self._lock = threading.Lock()
        self._stop = threading.Event()

        # State uses Linux-style event codes as emitted by `inputs`.
        self._abs = {
            "ABS_X": 0,      # left stick X
            "ABS_Y": 0,      # left stick Y
            "ABS_RX": 0,     # right stick X
            "ABS_RY": 0,     # right stick Y
            "ABS_Z": 0,      # left trigger (0..255 typically)
            "ABS_RZ": 0,     # right trigger (0..255 typically)
            "ABS_HAT0X": 0,  # dpad x (-1,0,1)
            "ABS_HAT0Y": 0,  # dpad y (-1,0,1)
        }
        self._key = {
            "BTN_SOUTH": 0,
            "BTN_EAST": 0,
            "BTN_NORTH": 0,
            "BTN_WEST": 0,
            "BTN_TL": 0,
            "BTN_TR": 0,
            "BTN_SELECT": 0,
            "BTN_START": 0,
            "BTN_THUMBL": 0,
            "BTN_THUMBR": 0,
            "BTN_MODE": 0,
        }

        self._thread = threading.Thread(target=self._reader_loop, name="physical_gamepad_reader", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()

    def _reader_loop(self):
        while not self._stop.is_set():
            try:
                events = self.device.read()  # blocks until at least one event
            except Exception:
                time.sleep(0.01)
                continue

            with self._lock:
                for e in events:
                    code = getattr(e, "code", None)
                    state = getattr(e, "state", None)
                    if code is None:
                        continue
                    if getattr(e, "ev_type", None) == "Absolute":
                        if code in self._abs and state is not None:
                            self._abs[code] = int(state)
                    elif getattr(e, "ev_type", None) == "Key":
                        if code in self._key and state is not None:
                            self._key[code] = 1 if int(state) else 0

    @staticmethod
    def _clamp_i16(v: int) -> int:
        return max(-32768, min(32767, int(v)))

    @staticmethod
    def _clamp_u8(v: int) -> int:
        return max(0, min(255, int(v)))

    def to_action(self):
        with self._lock:
            abs_state = dict(self._abs)
            key_state = dict(self._key)

        a = zero_action.copy()

        # Sticks: `inputs` usually reports signed 16-bit values for sticks.
        a["AXIS_LEFTX"] = np.array([self._clamp_i16(abs_state["ABS_X"])], dtype=np.long)
        a["AXIS_LEFTY"] = np.array([self._clamp_i16(-abs_state["ABS_Y"])], dtype=np.long)
        a["AXIS_RIGHTX"] = np.array([self._clamp_i16(abs_state["ABS_RX"])], dtype=np.long)
        a["AXIS_RIGHTY"] = np.array([self._clamp_i16(-abs_state["ABS_RY"])], dtype=np.long)

        # Triggers: `inputs` usually reports 0..255.
        a["LEFT_TRIGGER"] = np.array([self._clamp_u8(abs_state["ABS_Z"])], dtype=np.long)
        a["RIGHT_TRIGGER"] = np.array([self._clamp_u8(abs_state["ABS_RZ"])], dtype=np.long)

        # D-pad hats.
        hat_x = int(abs_state["ABS_HAT0X"])
        hat_y = int(abs_state["ABS_HAT0Y"])
        a["DPAD_LEFT"] = 1 if hat_x < 0 else 0
        a["DPAD_RIGHT"] = 1 if hat_x > 0 else 0
        a["DPAD_UP"] = 1 if hat_y < 0 else 0
        a["DPAD_DOWN"] = 1 if hat_y > 0 else 0

        # Buttons (common Xbox/DS4 codes).
        a["SOUTH"] = int(key_state["BTN_SOUTH"])
        a["EAST"] = int(key_state["BTN_EAST"])
        a["NORTH"] = int(key_state["BTN_NORTH"])
        a["WEST"] = int(key_state["BTN_WEST"])
        a["LEFT_SHOULDER"] = int(key_state["BTN_TL"])
        a["RIGHT_SHOULDER"] = int(key_state["BTN_TR"])
        a["BACK"] = int(key_state["BTN_SELECT"])
        a["START"] = int(key_state["BTN_START"])
        a["LEFT_THUMB"] = int(key_state["BTN_THUMBL"])
        a["RIGHT_THUMB"] = int(key_state["BTN_THUMBR"])
        a["GUIDE"] = int(key_state["BTN_MODE"])

        return a


policy = ModelClient(port=args.port)
policy.reset()
policy_info = policy.info()
action_downsample_ratio = policy_info["action_downsample_ratio"]

CKPT_NAME = Path(policy_info["ckpt_path"]).stem
NO_MENU = not args.allow_menu

PATH_DEBUG = PATH_REPO / "debug"
PATH_DEBUG.mkdir(parents=True, exist_ok=True)

PATH_OUT = (PATH_REPO / "out" / CKPT_NAME).resolve()
PATH_OUT.mkdir(parents=True, exist_ok=True)

BUTTON_PRESS_THRES = 0.5

# Find in path_out the list of existing video files, named 0001.mp4, 0002.mp4, etc.
# If they exist, find the max number and set the next number to be max + 1
video_files = sorted(PATH_OUT.glob("*_DEBUG.mp4"))
if video_files:
    existing_numbers = [f.name.split("_")[0] for f in video_files]
    existing_numbers = [int(n) for n in existing_numbers if n.isdigit()]
    next_number = max(existing_numbers) + 1
else:
    next_number = 1

PATH_MP4_DEBUG = PATH_OUT / f"{next_number:04d}_DEBUG.mp4"
PATH_MP4_CLEAN = PATH_OUT / f"{next_number:04d}_CLEAN.mp4"
PATH_ACTIONS = PATH_OUT / f"{next_number:04d}_ACTIONS.json"

import numpy as np
import cv2
import threading
from queue import Queue

def background_save_worker(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        path, image_np = item
        Image.fromarray(image_np).save(path)
        queue.task_done()

save_queue = Queue(maxsize=1)
save_thread = threading.Thread(target=background_save_worker, args=(save_queue,), daemon=True)
save_thread.start()

def preprocess_img(main_image):
    # Now main_image is already a numpy array from GamepadEnv.render()
    if main_image.shape[0] != 256 or main_image.shape[1] != 256:
        return cv2.resize(main_image, (256, 256), interpolation=cv2.INTER_AREA)
    return main_image

zero_action = OrderedDict(
        [ 
            ("WEST", 0),
            ("SOUTH", 0),
            ("BACK", 0),
            ("DPAD_DOWN", 0),
            ("DPAD_LEFT", 0),
            ("DPAD_RIGHT", 0),
            ("DPAD_UP", 0),
            ("GUIDE", 0),
            ("AXIS_LEFTX", np.array([0], dtype=np.long)),
            ("AXIS_LEFTY", np.array([0], dtype=np.long)),
            ("LEFT_SHOULDER", 0),
            ("LEFT_TRIGGER", np.array([0], dtype=np.long)),
            ("AXIS_RIGHTX", np.array([0], dtype=np.long)),
            ("AXIS_RIGHTY", np.array([0], dtype=np.long)),
            ("LEFT_THUMB", 0),
            ("RIGHT_THUMB", 0),
            ("RIGHT_SHOULDER", 0),
            ("RIGHT_TRIGGER", np.array([0], dtype=np.long)),
            ("START", 0),
            ("EAST", 0),
            ("NORTH", 0),
        ]
    )

TOKEN_SET = BUTTON_ACTION_TOKENS

print("Model loaded, starting environment...")
for i in range(3):
    print(f"{3 - i}...")
    time.sleep(1)

env = GamepadEnv(
    game=args.process,
    # image_width=256,
    # image_height=256,
    game_speed=1.0,
    env_fps=60,
    async_mode=True,
)

# Optional manual forwarding setup (physical controller -> NitroGen virtual controller).
vk_toggle = _vk_code_from_name(args.manual_hotkey)
if vk_toggle is None:
    raise ValueError(f"Unsupported --manual-hotkey '{args.manual_hotkey}'. Use e.g. F8, F9, ESC, TAB, SPACE.")

manual_mode = bool(args.manual)
_hotkey_prev_down = False
forwarder = None
try:
    forwarder = PhysicalGamepadForwarder(controller_index=0)
    print(f"Physical controller detected for manual forwarding: {forwarder.device_name!r}")
    if manual_mode:
        print(f"Manual mode ON (physical -> virtual). Press {args.manual_hotkey} to toggle back to model.")
    else:
        print(f"Manual mode OFF (model). Press {args.manual_hotkey} to toggle to manual forwarding.")
except Exception as e:
    if manual_mode:
        raise
    print(f"Warning: manual forwarding unavailable ({e}). Model-only mode will run.")

# These games requires to open a menu to initialize the controller
if args.process == "isaac-ng.exe":
    print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
    input("Press enter to create a virtual controller and start rollouts...")
    for i in range(3):
        print(f"{3 - i}...")
        time.sleep(1)

    def press(button):
        env.gamepad_emulator.press_button(button)
        env.gamepad_emulator.gamepad.update()
        time.sleep(0.05)
        env.gamepad_emulator.release_button(button)
        env.gamepad_emulator.gamepad.update()

    press("SOUTH")
    for k in range(5):
        press("EAST")
        time.sleep(0.3)

if args.process == "Cuphead.exe":
    print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
    # input("Press enter to create a virtual controller and start rollouts...")
    for i in range(3):
        print(f"{3 - i}...")
        time.sleep(1)

    def press(button):
        env.gamepad_emulator.press_button(button)
        env.gamepad_emulator.gamepad.update()
        time.sleep(0.05)
        env.gamepad_emulator.release_button(button)
        env.gamepad_emulator.gamepad.update()

    press("SOUTH")
    for k in range(5):
        press("EAST")
        time.sleep(0.3)

env.reset()
env.pause()


# Initial call to get state
obs_full, reward, terminated, truncated, info = env.step(action=zero_action)

frames = None
step_count = 0

class DummyRecorder:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def add_frame(self, frame): pass

SAVE_VIDEO = False

with (VideoRecorder(str(PATH_MP4_DEBUG), fps=60, crf=32, preset="medium") if SAVE_VIDEO else DummyRecorder()) as debug_recorder:
    with (VideoRecorder(str(PATH_MP4_CLEAN), fps=60, crf=28, preset="medium") if SAVE_VIDEO else DummyRecorder()) as clean_recorder:
        try:
            while True:
                # Hotkey toggle (edge-triggered).
                hotkey_down = (ctypes.windll.user32.GetAsyncKeyState(vk_toggle) & 0x8000) != 0
                if hotkey_down and not _hotkey_prev_down:
                    manual_mode = not manual_mode
                    print(f"Toggled control mode -> {'MANUAL (physical -> virtual)' if manual_mode else 'MODEL'}")
                _hotkey_prev_down = hotkey_down

                obs_model = preprocess_img(obs_full)
                # Save debug image in background
                if save_queue.empty():
                    save_queue.put((PATH_DEBUG / "latest.png", obs_model))

                if manual_mode and forwarder is not None:
                    # Manual: forward physical controller state into NitroGen's virtual controller.
                    a = forwarder.to_action()
                    if NO_MENU:
                        a["GUIDE"] = 0
                        a["START"] = 0
                        a["BACK"] = 0

                    obs_full, reward, terminated, truncated, info = env.step(action=a)

                    if SAVE_VIDEO:
                        obs_viz = np.array(obs_full).copy()
                        clean_viz = cv2.resize(obs_viz, (1920, 1080), interpolation=cv2.INTER_AREA)
                        debug_viz = cv2.resize(obs_viz, (1280, 720), interpolation=cv2.INTER_AREA)
                        debug_recorder.add_frame(debug_viz)
                        clean_recorder.add_frame(clean_viz)

                    step_count += 1
                    continue

                pred = policy.predict(obs_model)

                j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]

                n = len(buttons)
                assert n == len(j_left) == len(j_right), "Mismatch in action lengths"


                env_actions = []

                for i in range(n):
                    move_action = zero_action.copy()

                    xl, yl = j_left[i]
                    xr, yr = j_right[i]
                    move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.long)
                    move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.long)
                    move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.long)
                    move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.long)
                    
                    button_vector = buttons[i]
                    assert len(button_vector) == len(TOKEN_SET), "Button vector length does not match token set length"

                    
                    for name, value in zip(TOKEN_SET, button_vector):
                        if "TRIGGER" in name:
                            move_action[name] =  np.array([value * 255], dtype=np.long)
                        else:
                            move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0


                    env_actions.append(move_action)

                print(f"Executing {len(env_actions)} actions, each action will be repeated {action_downsample_ratio} times")

                for i, a in enumerate(env_actions):
                    if NO_MENU:
                        if a["START"]:
                            print("Model predicted start, disabling this action")
                        a["GUIDE"] = 0
                        a["START"] = 0
                        a["BACK"] = 0

                    for _ in range(action_downsample_ratio):
                        obs_full, reward, terminated, truncated, info = env.step(action=a)

                        if SAVE_VIDEO:
                            # resize obs to 720p
                            obs_viz = np.array(obs_full).copy()
                            clean_viz = cv2.resize(obs_viz, (1920, 1080), interpolation=cv2.INTER_AREA)
                            debug_viz = create_viz(
                                cv2.resize(obs_viz, (1280, 720), interpolation=cv2.INTER_AREA), # 720p
                                i,
                                j_left,
                                j_right,
                                buttons,
                                token_set=TOKEN_SET
                            )
                            debug_recorder.add_frame(debug_viz)
                            clean_recorder.add_frame(clean_viz)

                # Append env_actions dictionnary to JSONL file
                with open(PATH_ACTIONS, "a") as f:
                    for i, a in enumerate(env_actions):
                        # convert numpy arrays to lists for JSON serialization
                        for k, v in a.items():
                            if isinstance(v, np.ndarray):
                                a[k] = v.tolist()
                        a["step"] = step_count
                        a["substep"] = i
                        json.dump(a, f)
                        f.write("\n")


                step_count += 1
        finally:
            env.unpause()
            if forwarder is not None:
                forwarder.close()
            env.close()
