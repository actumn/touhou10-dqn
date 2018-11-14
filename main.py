
from th10.process import find_process
from th10.memory_reader import MemoryReader

th10_pid = find_process('th10.exe')
reader = MemoryReader(th10_pid)
player = reader.player_info()
print(player)
