from pynput.mouse import Controller
import time

mouse = Controller()
while True:
    mouse.move(100, 100)
    time.sleep(60)
    mouse.move(-100, -100)
    time.sleep(60)