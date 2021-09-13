#!/usr/bin/env python3
 
from ppadb.client import Client
from PIL import Image
import numpy
import time
import random
from os import system as terminal
 
 
 
 
terminal("cls")
adb = Client(host='127.0.0.1', port=5037)
devices = adb.devices()
 
if len(devices) == 0:
    print('no device attached')
    quit()
 
device = devices[0] 
    
    
image = device.screencap()




with open('./screenshots/screen.png', 'wb') as f:
    f.write(image)

photo = Image.open('./screenshots/screen.png')
rgbphoto = photo.convert('RGB')
width = photo.size[0]
height = photo.size[1]
print(height, width)

#print("Pokestop: " + str(photo.getpixel((525, 2045))))
#print("Team Rocket: " +  str(photo.getpixel((743, 1652))))    
#print("Pokeball: " + str(photo.getpixel(((938, 1958)))))
#print("OK: " + str(photo.getpixel(((540, 1450)))))

#time.sleep(1)
randx = random.randint(320,770)
randy = random.randint(1280,1683)
device.shell(f'input tap {int(randx)} {int(randy)}')
#time.sleep(1)




RGB = rgbphoto.getpixel((100,100))


print(RGB[0], RGB[1], RGB[2])

