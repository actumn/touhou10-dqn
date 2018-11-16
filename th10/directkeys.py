# direct inputs
# source to this solution and code:
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

import ctypes

DIK_Z = 0x2C
DIK_LEFT = 0xCB
DIK_RIGHT = 0xCD
DIK_UP = 0xC8
DIK_DOWN = 0xD0
DIK_LSHIFT = 0x2A
DIK_LCONTROL = 0x1D

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008
EXTRA = ctypes.c_ulong(0)

SendInput = ctypes.windll.user32.SendInput

PUL = ctypes.POINTER(ctypes.c_ulong)


# https://docs.microsoft.com/en-us/windows/desktop/api/winuser/ns-winuser-tagkeybdinput
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


# https://docs.microsoft.com/en-us/windows/desktop/api/winuser/ns-winuser-taghardwareinput
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


# https://docs.microsoft.com/en-us/windows/desktop/api/winuser/ns-winuser-tagmouseinput
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# SendInput: https://docs.microsoft.com/en-us/windows/desktop/api/winuser/nf-winuser-sendinput
def press_key(hex_key_code):
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, KEYEVENTF_SCANCODE, 0, ctypes.pointer(EXTRA))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def release_key(hex_key_code):
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(EXTRA))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

