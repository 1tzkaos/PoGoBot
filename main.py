#!/usr/bin/env python3
 
from ppadb.client import Client
from PIL import Image
import numpy
import time
import random
from os import system as terminal
import ctypes
from pypresence import Presence
 
 
 
 
terminal("cls")
adb = Client(host='127.0.0.1', port=5037)
devices = adb.devices()
 
if len(devices) == 0:
    print('no device attached')
    quit()
 
device = devices[0]
 
 
#
# Reset Stats
#
 
pokestops=0
caught=0
thrown=0
 
#
#


#client_id = '886711064132194354'  # Fake ID, put your real one here
#RPC = Presence(client_id,pipe=0)  # Initialize the client class
#RPC.connect() # Start the handshake loop
#start_time=time.time()
#RPC.update(large_image="pokeball", large_text="Pokemon Bot", start=start_time)
 
print("Starting bot!")
while 1:
 
 
 
 
    image = device.screencap()
 
 
 
 
    with open('./screenshots/screen.png', 'wb') as f:
        f.write(image)
 
    photo = Image.open('./screenshots/screen.png')
    photo = photo.convert('RGB')
    width = photo.size[0]
    height = photo.size[1]
 
 
    #print("Pokestop: " + str(photo.getpixel((525, 2045))))
    #print("Team Rocket: " +  str(photo.getpixel((743, 1652))))    
    #print("Pokeball: " + str(photo.getpixel(((938, 1958)))))
    #print("OK: " + str(photo.getpixel(((540, 1450)))))
 


    randx = random.randint(320,770)
    randy = random.randint(1280,1683)
    device.shell(f'input tap {int(randx)} {int(randy)}')

    pokestop = photo.getpixel((529, 2020))
    gym = photo.getpixel((900,2100))
    rando = photo.getpixel((941,2051))
    pokemon = photo.getpixel((938, 1958))
    pokecaught = photo.getpixel((540, 1450))
    newpokecaught = photo.getpixel((541, 1510))
    rando1 = photo.getpixel((886, 2156))
    teamrocket = photo.getpixel((743, 1652))
    menubutton = photo.getpixel((923,2019))
    #print("Caught: " + str(pokecaught) + "     Pokestop: " + str(pokestop) + "     Found: " + str(pokemon) + "     Menu: " + str(menubutton))

    if(rando[0]==26 and rando[1]==128 and rando[2]==145):
        time.sleep(1)
        device.shell(f'input tap 941 2051')
        time.sleep(1)
    if(gym[0]==33 and gym[1]==101 and gym[2]==218):
        print('Gym Found!')
        time.sleep(1)
        device.shell(f'input touchscreen swipe 200 1000 800 100 100')
        time.sleep(1)
        device.shell(f'input tap 532 2048')
        time.sleep(1)  
        device.shell(f'input touchscreen swipe 200 1000 800 100 100')
        time.sleep(1)
        device.shell(f'input tap 525 2045')
        time.sleep(1)
        device.shell(f'input tap 525 2045')

    if(menubutton[0]>24 and menubutton[0]<32 and menubutton[1]>130 and menubutton[1]<140 and menubutton[2]>145 and menubutton[2]<155):
        print("Menu button")
        device.shell(f'input tap 927 2021')
        time.sleep(1)
        device.shell(f'input tap 851 1651')
        time.sleep(0.5)
        device.shell(f'input tap 851 1651')
        time.sleep(1.5)
        star = photo.getpixel((232,1427))

        if(star==(255,213,122)):
            print(f"Good IV's")
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(1)
            device.shell(f'input tap 525 2045')
        else:
            print(f"Bad IV's")
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(1)
            device.shell(f'input tap 927 2021')
            time.sleep(1)
            device.shell(f'input tap 935 1848')
            time.sleep(1)
            device.shell(f'input tap 540 1250')
            time.sleep(0.5)
    

    if(pokemon[0]>205 and pokemon[0]<220 and pokemon[1]>50 and pokemon[1]<75 and pokemon[2]>25 and pokemon[2]<32):
        print("Pokemon Found!")
        throw = random.randint(10,150)
        thrown = thrown+1
        ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Pokeballs Thrown: {int(thrown)}")
        #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
        time.sleep(1)        
        print(f'Throwing at {int(throw)} power!')   
        device.shell(f'input touchscreen swipe 555 1900 555 1200 {int(throw)}')
        time.sleep(14)



    if(pokecaught[0]==115 and pokecaught[1]==214 and pokecaught[2]==157):
        print('Pokemon Caught!')
        caught = caught+1
        ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Pokeballs Thrown: {int(thrown)}")
        #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
        device.shell(f'input tap 540 1500')
        time.sleep(3)
        print("Menu button")
        device.shell(f'input tap 927 2021')
        time.sleep(1)
        device.shell(f'input tap 851 1651')
        time.sleep(0.5)
        device.shell(f'input tap 851 1651')
        time.sleep(1.5)
        star = photo.getpixel((232,1427))

        if(star==(255,213,122)):
            print(f"Good IV's")
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(1)
            device.shell(f'input tap 525 2045')
        else:
            print(f"Bad IV's")
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(1)
            device.shell(f'input tap 927 2021')
            time.sleep(1)
            device.shell(f'input tap 935 1848')
            time.sleep(1)
            device.shell(f'input tap 540 1250')
            time.sleep(0.5)



    if(pokecaught[0]==176 and pokecaught[1]==234 and pokecaught[2]==197):
        print('Pokemon Caught!')
        caught = caught+1
        ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Pokeballs Thrown: {int(thrown)}")
        #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
        device.shell(f'input tap 540 1500')
        time.sleep(3)
        print("Menu button")
        device.shell(f'input tap 927 2021')
        time.sleep(1)
        device.shell(f'input tap 851 1651')
        time.sleep(0.5)
        device.shell(f'input tap 851 1651')
        time.sleep(1.5)
        star = photo.getpixel((232,1427))

        if(star==(255,213,122)):
            print(f"Good IV's")
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(1)
            device.shell(f'input tap 525 2045')
        else:
            print(f"Bad IV's")
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(1)
            device.shell(f'input tap 927 2021')
            time.sleep(1)
            device.shell(f'input tap 935 1848')
            time.sleep(1)
            device.shell(f'input tap 540 1250')
            time.sleep(1)
    if(newpokecaught[0]==114 and newpokecaught[1]==214 and newpokecaught[2]==157):
        print('Pokemon Caught! New Pokemon!')
        caught = caught+1
        ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Pokeballs Thrown: {int(thrown)}")
        #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
        device.shell(f'input tap 540 1500')
        time.sleep(20)
        print("Menu button")
        device.shell(f'input tap 927 2021')
        time.sleep(2)
        device.shell(f'input tap 851 1651')
        time.sleep(2)
        device.shell(f'input tap 851 1651')
        time.sleep(1.5)
        star = photo.getpixel((232,1427))

        if(star==(255,213,122)):
            print(f"Good IV's")
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(1)
            device.shell(f'input tap 525 2045')
        else:
            print(f"Bad IV's")
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(1)
            device.shell(f'input tap 927 2021')
            time.sleep(1)
            device.shell(f'input tap 935 1848')
            time.sleep(1)
            device.shell(f'input tap 540 1250')
            time.sleep(1)



    if(rando[0]==49 and rando[1]==222 and rando1[2]==255):
        device.shell(f'input touchscreen swipe 200 1000 800 100 100')
        time.sleep(1)
        device.shell(f'input tap 525 2045')
        time.sleep(2)
    if(rando1[0]==31 and rando1[1]==143 and rando1[2]==249):
        device.shell(f'input touchscreen swipe 200 1000 800 100 100')
        time.sleep(1)
        device.shell(f'input tap 525 2045')
        time.sleep(2)


    if(pokestop[0]>218 and pokestop[1]>235 and pokestop[2]>225):
        print("Pokestop Found!")
        pokestops = pokestops+1
        ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Pokeballs Thrown: {int(thrown)}")
        #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
        device.shell(f'input touchscreen swipe 200 1000 800 100 100')
        time.sleep(0.2)
        device.shell(f'input tap 525 2045')
        time.sleep(1)




    
    if(teamrocket[0]==66 and teamrocket[1]==208 and teamrocket[2]==165):
        time.sleep(1)
        print("Team Rocket, Fuck that")
        device.shell(f'input tap 525 2045')
        time.sleep(1)

 
 
    '''
        elif(R==80 and G==255 and B==255):
            time.sleep(1)
            print(f"Pokestop Found at {int(x)} {int(y)}!")
            device.shell(f'input tap {int(x)} {int(y-10)}')
            time.sleep(1)
            device.shell(f'input touchscreen swipe 200 1000 800 100 100')
            time.sleep(1)
            device.shell(f'input tap 525 2045')
            time.sleep(5)
            break
    '''

