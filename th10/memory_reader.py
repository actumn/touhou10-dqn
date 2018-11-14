from ctypes import windll, c_ulong, byref, create_string_buffer, sizeof
from struct import unpack
from .data import *

OpenProcess = windll.kernel32.OpenProcess
ReadProcessMemory = windll.kernel32.ReadProcessMemory
CloseHandle = windll.kernel32

PROCESS_ALL_ACCESS = (0x000F0000 | 0x00100000 | 0xFFF)

buf_dword = create_string_buffer(4)
buf_dword64 = create_string_buffer(8)
bytes_read = c_ulong(0)


def read_buffer(process_handle, address, buffer):
    if windll.kernel32.ReadProcessMemory(process_handle,
                                         address, buffer, sizeof(buffer), byref(bytes_read)):
        return buffer
    else:
        raise MemoryError()


class MemoryReader(object):

    def __init__(self, pid):
        self.pid = pid
        self.process_handle = OpenProcess(PROCESS_ALL_ACCESS, False, pid)

    def __exit__(self):
        CloseHandle(self.pid)

    def read_int(self, address):
        buffer = read_buffer(self.process_handle, address, buf_dword)
        return unpack('i', buffer)[0]

    def read_uint(self, address):
        buffer = read_buffer(self.process_handle, address, buf_dword)
        return unpack('l', buffer)[0]

    def read_float(self, address):
        buffer = read_buffer(self.process_handle, address, buf_dword)
        return unpack('f', buffer)[0]

    # python porting from https://github.com/binvec/TH10_DataReversing/blob/master/TH10_DataReversing/data.cpp
    def bullet_info(self):
        bullets = []
        base_addr = self.read_int(0x004776F0)
        if base_addr == 0:
            return bullets

        ebx = base_addr + 0x60
        for i in range(2000):
            edi = ebx + 0x400
            bp = 0x0000FFFF & self.read_int(edi + 0x46)
            if bp:
                eax = self.read_int(0x00477810)
                if eax:
                    eax = self.read_int(eax + 0x58)
                    if not eax & 0x00000400:
                        x = self.read_float(ebx + 0x3B4)
                        y = self.read_float(ebx + 0x3B8)
                        w = self.read_float(ebx + 0x3F0)
                        h = self.read_float(ebx + 0x3F4)
                        dx = self.read_float(ebx + 0x3C0)
                        dy = self.read_float(ebx + 0x3C4)
                        self.bullets += [
                            GameObject(Vec2d(x, y), Vec2d(dx, dy), w, h)
                        ]

            ebx += 0x07F0

        return bullets

    def enemies_info(self):
        enemies = []
        base_addr = self.read_int(0x00477704)
        if base_addr == 0:
            return enemies

        base_addr = self.read_int(base_addr + 0x58)
        if base_addr:
            while True:
                obj_addr = self.read_int(base_addr)
                obj_next = self.read_int(base_addr + 0x4)
                obj_addr += 0x103C
                t = self.read_uint(base_addr + 0x1444)
                if t & 0x52 == 0:
                    x = self.read_float(base_addr + 0x2C)
                    y = self.read_float(base_addr + 0x30)
                    w = self.read_float(base_addr + 0xB8)
                    h = self.read_float(base_addr + 0xBC)
                    dx = self.read_float(base_addr + 0x38)
                    dy = self.read_float(base_addr + 0x3C)
                    self.enemies += [
                        GameObject(Vec2d(x, y), Vec2d(dx, dy), w, h)
                    ]
                if obj_next == 0:
                    break

                base_addr = obj_next

        return enemies

    def items_info(self):
        items = []
        base_addr = self.read_int(0x00477818)
        if base_addr == 0:
            return items

        esi = base_addr + 0x14
        ebp = esi + 0x3B0
        for i in range(2000):
            eax = self.read_int(ebp + 0x2C)
            """
            The types of points are divided into the following types
            eax == 0: not existing
            eax == 1: normal receivable point
            eax == 2: points that Boom have made while deleting bullets (putting B)
            eax == 3: points that arrive at the collection line, put B, etc
            eax == 4: points of the collection range of the arrival, automatic recovery
            """
            if eax == 1:
                x = self.read_float(ebp - 0x4)
                y = self.read_float(ebp)
                dx = self.read_float(ebp + 0x8)
                dy = self.read_float(ebp + 0xC)
                item_type = self.read_int(ebp + 0x30)
                """
                The types of normal points are divided into the following types
                type == 1: Power
                type == 2: Point
                type == 3: Faith
                type == 4: Large Power
                type == 5: Large point
                type == 6: Unknown
                type == 7: Life Items
                type == 8: Point
                type == 9: Faith, converted from Power when full of power
                type == 10: Power, dropped by Boss
                There is no width and height at the point. It will be automatically charged 
                when the player is close to the point, set to 6 for convenient display
                """
                self.items += [
                    Item(p=Vec2d(x, y), v=Vec2d(dx, dy),
                         w=6, h=6, item_type=item_type),
                ]

            ebp += 0x3F0

        return items

    # this project will only use here
    def player_info(self):
        base_addr = self.read_int(0x00477834)
        if base_addr == 0:
            return Player(p=Vec2d(0, 400), h=0, w=0)

        p = Vec2d(x=self.read_float(base_addr + 0x3C0),
                  y=self.read_float(base_addr + 0x3C4))
        v = Vec2d(x=self.read_float(base_addr + 0x3F0) / 100.,
                  y=self.read_float(base_addr + 0x3F4) / 100.)
        w = self.read_float(base_addr + 0x41C) * 2
        h = w
        slow = self.read_float(base_addr + 0x4474)
        powers = self.read_int(0x00474C48) / 20.
        player_type = self.read_float(0x00474C68)
        item_obtain_range = self.read_float(0x00476FB0) + player_type * 4
        if slow:
            item_obtain_range *= 2.5

        life = self.read_int(0x00474C70) + 1
        player_status = self.read_int(base_addr + 0x458)
        invincible_time = self.read_int(base_addr + 0x4310)
        return Player(p=p, v=v, w=w, h=h,
                      powers=powers, life=life, player_type=player_type, slow=slow,
                      item_obtain_range=item_obtain_range, player_status=player_status, invincible_time=invincible_time)

    def lasers_info(self):
        lasers = []
        base_addr = self.read_int(0x0047781C)
        if base_addr == 0:
            return lasers

        obj_addr = self.read_int(base_addr + 0x18)
        if obj_addr:
            while True:
                obj_next = self.read_int(obj_addr + 0x8)
                x = self.read_int(obj_addr + 0x24)
                y = self.read_int(obj_addr + 0x28)
                arc = self.read_int(obj_addr + 0x3C)
                h = self.read_int(obj_addr + 0x40)
                w = self.read_int(obj_addr + 0x44)
                dx = self.read_int(obj_addr + 0x30)
                dy = self.read_int(obj_addr + 0x34)
                lasers += [
                    Laser(p=Vec2d(x, y), v=Vec2d(dx, dy),
                          w=w, h=h, arc=arc)
                ]
                if obj_next == 0:
                    break
                obj_addr = obj_next

        return lasers

