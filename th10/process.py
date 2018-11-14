import win32gui
import win32process
import psutil


def find_process(process_name):
    def callback(handle, result):
        if win32gui.IsWindowVisible(handle) and win32gui.IsWindowEnabled(handle):
            thread_id, process_id = win32process.GetWindowThreadProcessId(handle)
            process = psutil.Process(process_id)
            if process.name() == process_name:
                result.append(process_id)
        return True

    process_ids = []
    win32gui.EnumWindows(callback, process_ids)
    return process_ids[0]

# win32gui.SendMessage(hwnd, win32con.WM_CLOSE, 0, 0)
