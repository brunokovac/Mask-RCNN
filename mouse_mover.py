from pynput.mouse import Controller
import time
import random

time.sleep(20)

mouse = Controller()
current_x, current_y = mouse.position

max_x = current_x + 100
max_y = current_y + 100

min_x = current_x - 100
min_y = current_y - 100

while True:
    x = random.randint(min_x, max_x)
    y = random.randint(min_y, max_y)
    mouse.position = (x, y)
    time.sleep(random.randint(50, 70))