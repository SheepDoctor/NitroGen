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
import pygame
from PIL import Image

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO
from nitrogen.inference_viz import create_viz, VideoRecorder
from nitrogen.inference_client import ModelClient

import argparse
parser = argparse.ArgumentParser(description="VLM Inference")
# parser.add_argument("--process", type=str, default="celeste.exe", help="Game to play")
# parser.add_argument("--process", type=str, default="isaac-ng.exe", help="Game to play")
# parser.add_argument("--process", type=str, default="Cuphead.exe", help="Game to play")
# parser.add_argument("--process", type=str, default="NineSols.exe", help="Game to play")
parser.add_argument("--process", type=str, default="FC26.exe", help="Game to play")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu actions (Disabled by default)")
parser.add_argument("--port", type=int, default=5555, help="Port for model server")
parser.add_argument("--manual", default=True, action="store_true", help="Start in manual controller-forwarding mode")
parser.add_argument("--manual-hotkey", type=str, default="F12", help="Hotkey to toggle manual/model mode (default: F8)")
parser.add_argument("--switch", default=False, action="store_true", help="Use Nintendo Switch controller button layout")

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
    Read a physical gamepad using pygame and convert its current state into NitroGen's unified action dict.
    This supports Switch controllers and handles mapping to virtual Xbox controllers.
    """

    def __init__(self, controller_index: int = 0, nintendo_layout: bool = False):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No physical gamepads detected by 'pygame'. "
                               "If using HidHide, ensure Python is whitelisted.")
        
        self.joy = pygame.joystick.Joystick(controller_index)
        self.joy.init()
        self.device_name = self.joy.get_name()
        self.nintendo_layout = nintendo_layout

        self._lock = threading.Lock()
        self._stop = threading.Event()

        # Internal state
        self.axes = [0.0] * self.joy.get_numaxes()
        self.buttons = [0] * self.joy.get_numbuttons()
        self.hats = [(0, 0)] * self.joy.get_numhats()

        self._thread = threading.Thread(target=self._reader_loop, name="physical_gamepad_reader", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if hasattr(self, 'joy'):
            self.joy.quit()

    def _reader_loop(self):
        while not self._stop.is_set():
            pygame.event.pump()
            with self._lock:
                for i in range(len(self.axes)):
                    self.axes[i] = self.joy.get_axis(i)
                for i in range(len(self.buttons)):
                    self.buttons[i] = self.joy.get_button(i)
                for i in range(len(self.hats)):
                    self.hats[i] = self.joy.get_hat(i)
            time.sleep(0.005)

    @staticmethod
    def _clamp_i16(v: float) -> int:
        return max(-32768, min(32767, int(v * 32767)))

    @staticmethod
    def _clamp_u8(v: float) -> int:
        # Map axis -1..1 to 0..255 or 0..1 to 0..255
        # For triggers, they often start at -1 (unpressed) and go to 1 (pressed)
        val = int(((v + 1) / 2) * 255) if v < 0 else int(v * 255)
        return max(0, min(255, val))

    def to_action(self):
        with self._lock:
            axes = list(self.axes)
            buttons = list(self.buttons)
            hats = list(self.hats)

        a = zero_action.copy()

        # 1. 摇杆 (标准 SDL2)
        if len(axes) > 1:
            a["AXIS_LEFTX"] = np.array([self._clamp_i16(axes[0])], dtype=np.int64)
            a["AXIS_LEFTY"] = np.array([self._clamp_i16(axes[1])], dtype=np.int64)
        if len(axes) > 3:
            a["AXIS_RIGHTX"] = np.array([self._clamp_i16(axes[2])], dtype=np.int64)
            a["AXIS_RIGHTY"] = np.array([self._clamp_i16(axes[3])], dtype=np.int64)

        # 2. 方向键 (hats)
        if hats:
            hx, hy = hats[0]
            a["DPAD_LEFT"] = 1 if hx < 0 else 0
            a["DPAD_RIGHT"] = 1 if hx > 0 else 0
            a["DPAD_UP"] = 1 if hy > 0 else 0
            a["DPAD_DOWN"] = 1 if hy < 0 else 0

        # 3. 主功能键 (A, B, X, Y)
        # Switch Pro 模式通常为: 0:B, 1:A, 2:Y, 3:X
        if self.nintendo_layout:
            # 这里的映射逻辑：让物理位置上的按键对应到虚拟控制器的逻辑位置
            # 如果是 Switch 布局，物理 A 在右侧，物理 B 在下方
            a["SOUTH"] = buttons[1] if len(buttons) > 1 else 0  # 物理 A (1) -> 虚拟 A (SOUTH)
            a["EAST"] = buttons[0] if len(buttons) > 0 else 0   # 物理 B (0) -> 虚拟 B (EAST)
            a["WEST"] = buttons[3] if len(buttons) > 3 else 0   # 物理 X (3) -> 虚拟 X (WEST)
            a["NORTH"] = buttons[2] if len(buttons) > 2 else 0  # 物理 Y (2) -> 虚拟 Y (NORTH)
        else:
            # 物理位置映射 (下方按键始终为 SOUTH/Accept)
            a["SOUTH"] = buttons[0] if len(buttons) > 0 else 0
            a["EAST"] = buttons[1] if len(buttons) > 1 else 0
            a["WEST"] = buttons[2] if len(buttons) > 2 else 0
            a["NORTH"] = buttons[3] if len(buttons) > 3 else 0
        
        # 设置冗余的 "RIGHT_*" 组合键
        a["RIGHT_BOTTOM"], a["RIGHT_RIGHT"] = a["SOUTH"], a["EAST"]
        a["RIGHT_LEFT"], a["RIGHT_UP"] = a["WEST"], a["NORTH"]

        # 4. 肩键 (LB, RB)
        # 在绝大多数 Switch/Xbox HID 模式下: 4 是 LB (L), 5 是 RB (R)
        if len(buttons) > 4: a["LEFT_SHOULDER"] = buttons[4]
        if len(buttons) > 5: a["RIGHT_SHOULDER"] = buttons[5]

        # 5. 扳机键 (LT, RT) 和其他辅助按键
        if len(axes) > 5:
            # Xbox 模式: 扳机是轴 4, 5. 
            # 按键: 6:BACK, 7:START, 8:LSB, 9:RSB, 10:GUIDE
            a["LEFT_TRIGGER"] = np.array([self._clamp_u8(axes[4])], dtype=np.int64)
            a["RIGHT_TRIGGER"] = np.array([self._clamp_u8(axes[5])], dtype=np.int64)
            if len(buttons) > 6: a["BACK"] = buttons[6]
            if len(buttons) > 7: a["START"] = buttons[7]
            if len(buttons) > 8: a["LEFT_THUMB"] = buttons[8]
            if len(buttons) > 9: a["RIGHT_THUMB"] = buttons[9]
            if len(buttons) > 10: a["GUIDE"] = buttons[10]
        else:
            # Switch 模式: 扳机 (ZL/ZR) 通常是按键 6, 7
            # 按键: 8:Minus, 9:Plus, 10:Home, 11:Capture, 12:LSB, 13:RSB
            if len(buttons) > 6: a["LEFT_TRIGGER"] = np.array([255 if buttons[6] else 0], dtype=np.int64)
            if len(buttons) > 7: a["RIGHT_TRIGGER"] = np.array([255 if buttons[7] else 0], dtype=np.int64)
            if len(buttons) > 8: a["BACK"] = buttons[8]
            if len(buttons) > 9: a["START"] = buttons[9]
            if len(buttons) > 10: a["GUIDE"] = buttons[10]
            if len(buttons) > 12: a["LEFT_THUMB"] = buttons[12]
            if len(buttons) > 13: a["RIGHT_THUMB"] = buttons[13]

        return a


policy = ModelClient(port=args.port)
policy.reset()
policy_info = policy.info()
action_downsample_ratio = policy_info["action_downsample_ratio"]

CKPT_NAME = Path(policy_info["ckpt_path"]).stem
NO_MENU = not args.allow_menu

PATH_DEBUG = PATH_REPO / "debug"
PATH_DEBUG.mkdir(parents=True, exist_ok=True)

PATH_OUT = (PATH_REPO / "out" / args.process).resolve()
PATH_OUT.mkdir(parents=True, exist_ok=True)

BUTTON_PRESS_THRES = 0.5

def get_next_paths(base_path):
    video_files = sorted(base_path.glob("*_DEBUG.mp4"))
    if video_files:
        existing_numbers = []
        for f in video_files:
            prefix = f.name.split("_")[0]
            if prefix.isdigit():
                existing_numbers.append(int(prefix))
        next_num = max(existing_numbers) + 1 if existing_numbers else 1
    else:
        next_num = 1
    
    return (
        base_path / f"{next_num:04d}_DEBUG.mp4",
        base_path / f"{next_num:04d}_CLEAN.mp4",
        base_path / f"{next_num:04d}_ACTIONS.json"
    )

# These will be updated dynamically when recording starts
PATH_MP4_DEBUG = None
PATH_MP4_CLEAN = None
PATH_ACTIONS = None

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
            ("AXIS_LEFTX", np.array([0], dtype=np.int64)),
            ("AXIS_LEFTY", np.array([0], dtype=np.int64)),
            ("LEFT_SHOULDER", 0),
            ("LEFT_TRIGGER", np.array([0], dtype=np.int64)),
            ("AXIS_RIGHTX", np.array([0], dtype=np.int64)),
            ("AXIS_RIGHTY", np.array([0], dtype=np.int64)),
            ("LEFT_THUMB", 0),
            ("RIGHT_THUMB", 0),
            ("RIGHT_SHOULDER", 0),
            ("RIGHT_TRIGGER", np.array([0], dtype=np.int64)),
            ("START", 0),
            ("EAST", 0),
            ("NORTH", 0),
            ("RIGHT_BOTTOM", 0),
            ("RIGHT_LEFT", 0),
            ("RIGHT_RIGHT", 0),
            ("RIGHT_UP", 0),
        ]
    )

TOKEN_SET = BUTTON_ACTION_TOKENS

print("Model loaded, starting environment...")
for i in range(3):
    print(f"{3 - i}...")
    time.sleep(1)

env = GamepadEnv(
    game=args.process,
    image_width=256,
    image_height=256,
    game_speed=1.0,
    env_fps=60,
    async_mode=True,
)

# Optional manual forwarding setup (physical controller -> NitroGen virtual controller).
vk_toggle = _vk_code_from_name(args.manual_hotkey)
if vk_toggle is None:
    raise ValueError(f"Unsupported --manual-hotkey '{args.manual_hotkey}'. Use e.g. F8, F9, ESC, TAB, SPACE.")

manual_mode = bool(args.manual)
should_record = False

def hotkey_listener():
    global manual_mode, should_record
    _prev_down = False
    while True:
        down = (ctypes.windll.user32.GetAsyncKeyState(vk_toggle) & 0x8000) != 0
        if down and not _prev_down:
            manual_mode = not manual_mode
            # Enable recording when switching to model mode (backend control)
            should_record = not manual_mode
            print(f"Toggled control mode -> {'MANUAL (physical -> virtual)' if manual_mode else 'MODEL'}")
            if should_record:
                print("Recording enabled - starting next frame.")
            else:
                print("Recording disabled - finishing video.")
        _prev_down = down
        time.sleep(0.01)

threading.Thread(target=hotkey_listener, daemon=True).start()

forwarder = None
try:
    forwarder = PhysicalGamepadForwarder(controller_index=0, nintendo_layout=args.switch)
    print(f"Physical controller detected for manual forwarding: {forwarder.device_name!r}")
    print(f"NOTE: If your controller is not detected, ensure HidHide whitelists: {sys.executable}")
    if manual_mode:
        print(f"Manual mode ON (physical -> virtual). Press {args.manual_hotkey} to toggle back to model.")
    else:
        print(f"Manual mode OFF (model). Press {args.manual_hotkey} to toggle to manual forwarding.")
except Exception as e:
    if manual_mode:
        raise
    print(f"Warning: manual forwarding unavailable ({e}). Model-only mode will run.")

# # These games requires to open a menu to initialize the controller
# if args.process == "isaac-ng.exe":
#     print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
#     input("Press enter to create a virtual controller and start rollouts...")
#     for i in range(3):
#         print(f"{3 - i}...")
#         time.sleep(1)

#     def press(button):
#         env.gamepad_emulator.press_button(button)
#         env.gamepad_emulator.gamepad.update()
#         time.sleep(0.05)
#         env.gamepad_emulator.release_button(button)
#         env.gamepad_emulator.gamepad.update()

#     press("SOUTH")
#     for k in range(5):
#         press("EAST")
#         time.sleep(0.3)

# if args.process == "Cuphead.exe":
#     print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
#     # input("Press enter to create a virtual controller and start rollouts...")
#     for i in range(3):
#         print(f"{3 - i}...")
#         time.sleep(1)

#     def press(button):
#         env.gamepad_emulator.press_button(button)
#         env.gamepad_emulator.gamepad.update()
#         time.sleep(0.05)
#         env.gamepad_emulator.release_button(button)
#         env.gamepad_emulator.gamepad.update()

#     press("SOUTH")
#     for k in range(5):
#         press("EAST")
#         time.sleep(0.3)

env.reset()
env.pause()


# Initial call to get state
obs_full, reward, terminated, truncated, info = env.step(action=zero_action)

frames = None
step_count = 0

debug_recorder = None
clean_recorder = None

try:
    while True:
        # Manage recording state
        if should_record and debug_recorder is None:
            PATH_MP4_DEBUG, PATH_MP4_CLEAN, PATH_ACTIONS = get_next_paths(PATH_OUT)
            debug_recorder = VideoRecorder(str(PATH_MP4_DEBUG), fps=60, crf=32, preset="medium")
            clean_recorder = VideoRecorder(str(PATH_MP4_CLEAN), fps=60, crf=28, preset="medium")
            print(f"Recording started: {PATH_MP4_DEBUG}")
        elif not should_record and debug_recorder is not None:
            debug_recorder.close()
            clean_recorder.close()
            debug_recorder = None
            clean_recorder = None
            print("Recording stopped.")

        obs_model = preprocess_img(obs_full)
        # Save debug image in background
        # if save_queue.empty():
        #     save_queue.put((PATH_DEBUG / "latest.png", obs_model))

        if manual_mode and forwarder is not None:
            # Manual: forward physical controller state into NitroGen's virtual controller.
            a = forwarder.to_action()
            if NO_MENU:
                a["GUIDE"] = 0
                a["START"] = 0
                a["BACK"] = 0

            obs_full, reward, terminated, truncated, info = env.step(action=a)

            if debug_recorder is not None:
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
            move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.int64)
            move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.int64)
            move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.int64)
            move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.int64)
            
            button_vector = buttons[i]
            assert len(button_vector) == len(TOKEN_SET), "Button vector length does not match token set length"

            
            for name, value in zip(TOKEN_SET, button_vector):
                if "TRIGGER" in name:
                    move_action[name] =  np.array([value * 255], dtype=np.int64)
                else:
                    move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0


            env_actions.append(move_action)

        # print(f"Executing {len(env_actions)} actions, each action will be repeated {action_downsample_ratio} times")

        for i, a in enumerate(env_actions):
            if manual_mode:
                break
            if NO_MENU:
                if a["START"]:
                    print("Model predicted start, disabling this action")
                a["GUIDE"] = 0
                a["START"] = 0
                a["BACK"] = 0

            for _ in range(action_downsample_ratio):
                if manual_mode:
                    break
                obs_full, reward, terminated, truncated, info = env.step(action=a)

                if debug_recorder is not None:
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
        # with open(PATH_ACTIONS, "a") as f:
        #     for i, a in enumerate(env_actions):
        #         # convert numpy arrays to lists for JSON serialization
        #         for k, v in a.items():
        #             if isinstance(v, np.ndarray):
        #                 a[k] = v.tolist()
        #         a["step"] = step_count
        #         a["substep"] = i
        #         json.dump(a, f)
        #         f.write("\n")


        step_count += 1
finally:
    env.unpause()
    if forwarder is not None:
        forwarder.close()
    if debug_recorder is not None:
        debug_recorder.close()
    if clean_recorder is not None:
        clean_recorder.close()
    env.close()
