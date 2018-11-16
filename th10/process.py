import win32gui
import win32process
import psutil
import operator
from PIL import ImageGrab


def set_foreground(hwnd):
    win32gui.SetForegroundWindow(hwnd)


def image_grab(hwnd, bbox):
    window_bbox = win32gui.GetWindowRect(hwnd)
    bbox = tuple(map(operator.add, bbox, window_bbox))
    return ImageGrab.grab(bbox)


def find_process(process_name):
    def callback(hwnd, result):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(process_id)
            if process.name() == process_name:
                result.append((process_id, hwnd))
        return True

    process_ids = []
    win32gui.EnumWindows(callback, process_ids)
    if len(process_ids) != 1:
        raise AssertionError(f"{process_name} not found")

    return process_ids[0][0], process_ids[0][1]


# win32gui.SendMessage(hwnd, win32con.WM_CLOSE, 0, 0)
