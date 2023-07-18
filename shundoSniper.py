#!/usr/bin/env python3

from ppadb.client import Client
from PIL import Image, ImageFile
import numpy
import time
import random
from os import system as terminal
import ctypes
from pypresence import Presence
from colorama import Fore, Back, Style


terminal("cls")
adb = Client(host='127.0.0.1', port=5037)
devices = adb.devices()
print(devices)

if len(devices) == 0:
    print('no device attached')
    quit()

device = devices[0]

ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# Reset Stats
#

shinies = 0
pokestops = 0
caught = 0
thrown = 0
attempts = 0

#
#


client_id = '886711064132194354'  # Fake ID, put your real one here
RPC = Presence(client_id, pipe=0)  # Initialize the client class
RPC.connect()  # Start the handshake loop
start_time = time.time()
RPC.update(large_image="pokeball",
           large_text=f"Pokemon Bot", start=start_time)

print("Starting bot!")
while 1:
    attempts += 1
    if (attempts >= 10):
        attempts = 0
        device.shell('input tap 800 2300')
        time.sleep(3)
        device.shell('input tap 559 1860')
        time.sleep(3)
        device.shell(f'input tap 537 952')
        time.sleep(3)

    print("Teleporting to possible SHUNDO")
    device.shell('input tap 50 300')
    wait = 0
    while wait < 15:
        try:
            image = device.screencap()
            print('Updating...')
            device.shell('input tap 540 1350')
            time.sleep(0.2)
            device.shell('input tap 540 1400')
            time.sleep(0.2)
            device.shell('input tap 540 1450')
            time.sleep(0.2)
            device.shell('input tap 540 1600')
            time.sleep(0.2)
            device.shell('input tap 540 1650')
            time.sleep(0.2)

        except:
            pass

        with open('./screenshots/screen.png', 'wb') as f:
            f.write(image)

        photo = Image.open('./screenshots/screen.png')
        photo = photo.convert('RGB')
        width = photo.size[0]
        height = photo.size[1]

        pokemon = photo.getpixel((938, 1958))
        pokestop = photo.getpixel((962, 273))
        homescreen = photo.getpixel((573, 2059))
        mainscreen = photo.getpixel((540, 2010))

        print(homescreen)
        if (homescreen[0] == 245 and homescreen[1] == 245 and homescreen[2] == 245):
            print("Home Screen Identified. Starting failsafe.")
            time.sleep(0.5)
            device.shell(f'input tap 537 952')

        if (pokemon[0] > 205 and pokemon[0] < 220 and pokemon[1] > 50 and pokemon[1] < 78 and pokemon[2] > 20 and pokemon[2] < 32):
            print("Pokemon Found!")
            throw = random.randint(10, 150)
            thrown = thrown+1
            ctypes.windll.kernel32.SetConsoleTitleW(
                f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Shinies: {int(shinies)} | Pokeballs Thrown: {int(thrown)} | XP Gained: {int(xp)} ")
            RPC.update(large_image="pokeball", large_text="Pokemon Bot",
                       details=f"PogoBot\nShinies Caught: {int(shinies)}", start=start_time)
            time.sleep(1)
            print(f'Throwing at {int(throw)} power!')
            device.shell(
                f'input touchscreen swipe 555 2100 555 1200 {int(throw)}')
            time.sleep(5)
            if (mainscreen[0] >= 245 and mainscreen[1] > 45 and mainscreen[1] < 70 and mainscreen[2] > 55 and mainscreen[2] < 80):
                shinies += 1
                print("SHUNDO CAUGHT")
                time.sleep(7200)
                break

        if (pokestop[0] >= 230 and pokestop[1] >= 240 and pokestop[2] >= 230):
            print("Pokestop Found :( Exiting)")
            device.shell(f'input tap 538 2045')
            time.sleep(0.5)
        wait += 1
        print(wait)
