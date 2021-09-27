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

shinies=0
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
 
    try: 
        image = device.screencap()
        print('Updating...')
    except:
        pass


 
 
 
 
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

    pokestop = photo.getpixel((962, 273))
    gym = photo.getpixel((900,2100))
    rando = photo.getpixel((941,2051))
    pokemon = photo.getpixel((938, 1958))
    pokecaught = photo.getpixel((540, 1450))
    newpokecaught = photo.getpixel((541, 1510))
    rando1 = photo.getpixel((886, 2156))
    teamrocket = photo.getpixel((743, 1652))
    menubutton = photo.getpixel((930,2000))
    failsafe = photo.getpixel((479,1616))
    homescreen = photo.getpixel((573,2059))
    battlefailsafe = photo.getpixel((290, 229))
    menufailsafe = photo.getpixel((544, 1865))
    xp = (caught*1120)+(pokestops*100)
    shop = photo.getpixel((546, 1175))
    battle = photo.getpixel((705, 1427))
    battleinfo = photo.getpixel((684, 1896))
    closet = photo.getpixel((977, 219))
    pokedex = photo.getpixel((1023, 2073))
    menu = photo.getpixel((965,412))
    shopfailsafe = photo.getpixel((270,1284))

    if(shopfailsafe[0]>250 and shopfailsafe[1]>119 and shopfailsafe[1]<125 and shopfailsafe[2]>145 and shopfailsafe[2]<153):
        print("In Shop")
        device.shell(f'input tap 550 2070')
        time.sleep(0.5)
        device.shell(f'input tap 550 2070')
        time.sleep(0.5)

    if(menu[0]>250 and menu[1]>90 and menu[1]<100 and menu[2]>20 and menu[2]<25):
        print("In Menu")
        device.shell(f'input tap 550 2070')
        time.sleep(0.5)

    #print("Caught: " + str(pokecaught) + "     Pokestop: " + str(pokestop) + "     Found: " + str(pokemon) + "     Menu: " + str(menubutton))
    if(pokedex[0]>145 and pokedex[0]<150 and pokedex[1]>115 and pokedex[1]<120 and pokedex[2]>250):
        print("In Pokedex")
        device.shell(f'input tap 550 2070')
        time.sleep(0.5)

    if(closet[0]>15 and closet[0]<25 and closet[1]>128 and closet[1]<136 and closet[2]>134 and closet[2]<142):
        print("In Closet")
        device.shell(f'input tap 539 2120')
        time.sleep(0.5)

    if(battleinfo[0]==14 and battleinfo[1]==42 and battleinfo[2]==33):
        print('Battle info screen')
        device.shell(f'input tap 533 2061')
        time.sleep(0.3)
        device.shell(f'input tap 533 2061')
        time.sleep(0.5)
    if(battle[0]>230 and battle[0]<235 and battle[1]>125 and battle[1]<130 and battle[2]>180 and battle[2]<187):
        print("Trying to battle in league")
        print("Returning to map")
        device.shell(f'input tap 543 2054') 
        time.sleep(1)



    if(shop[0]>220 and shop[0]<230 and shop[1]>195 and shop[1]<200 and shop[2]>62 and shop[2]<70):
        print('In Shop Menu')
        device.shell(f'input tap 547 2052') 
        time.sleep(1)

    if(menufailsafe[0]>100 and menufailsafe[0]<110 and menufailsafe[1]>210 and menufailsafe[1]<220 and menufailsafe[2]>150 and menufailsafe[2]<160):
        print("In battle")
        device.shell(f'input tap 547 2052') 
        time.sleep(1)
        device.shell(f'input tap 113 253') 
        time.sleep(1)
        device.shell(f'input tap 520 1116') 
        time.sleep(1)


    if(battlefailsafe[0]>230 and battlefailsafe[0]<240 and battlefailsafe[1]>238 and battlefailsafe[1]<242 and battlefailsafe[2]>230 and battlefailsafe[2]<240):
        print("In battle")
        device.shell(f'input tap 113 256') 
        time.sleep(1)
        device.shell(f'input tap 529 1144') 
        time.sleep(1)
# leave: 111 277

    if(homescreen[0]==87 and homescreen[1]==66 and homescreen[2]>249):
        print("Home Screen Identified. Starting failsafe.")
        time.sleep(0.5)
        device.shell(f'input tap 537 952')
        time.sleep(60)
        device.shell(f'input tap 537 1325')

    if(failsafe[0]== 215 and failsafe[1]==139 and failsafe[2]==255):
        print("Failsafe Triggered!")
        time.sleep(0.5)
        device.shell(f'input tap 543 2064')
        time.sleep(1)
        image = device.screencap()
        with open('./screenshots/screen.png', 'wb') as f:
            f.write(image)
        time.sleep(2)
        shiny = photo.getpixel((704, 544))
        if(shiny[0]>250 and shiny[1]>230 and shiny[2]<240 and shiny[2]<10):
            print(Fore.YELLOW + 'SHINY!!!!' + Fore.RESET)
            shinies = shinies+1
            device.shell(f'input tap 536 2061')
            time.sleep(0.5)
        else:
            print("Menu button")
            device.shell(f'input tap 927 2021')
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(0.5)
            device.shell(f'input tap 851 1651')
            time.sleep(1.5)
            image = device.screencap()
            with open('./screenshots/screen.png', 'wb') as f:
                f.write(image)
            time.sleep(0.5)
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

    if(pokemon[0]>205 and pokemon[0]<220 and pokemon[1]>50 and pokemon[1]<78 and pokemon[2]>20 and pokemon[2]<32):
        print("Pokemon Found!")
        throw = random.randint(10,150)
        thrown = thrown+1
        ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Shinies: {int(shinies)} | Pokeballs Thrown: {int(thrown)} | XP Gained: {int(xp)} ")
        #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
        time.sleep(1)        
        print(f'Throwing at {int(throw)} power!')   
        device.shell(f'input touchscreen swipe 555 2100 555 1200 {int(throw)}')
        time.sleep(14)



    if(pokecaught[0]==115 and pokecaught[1]==214 and pokecaught[2]==157):
        print('Pokemon Caught!')
        caught = caught+1
        ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Shinies: {int(shinies)} | Pokeballs Thrown: {int(thrown)} | XP Gained: {int(xp)} ")
        #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
        device.shell(f'input tap 540 1500')
        time.sleep(3)
        image = device.screencap()
        with open('./screenshots/screen.png', 'wb') as f:
            f.write(image)
        time.sleep(2)
        shiny = photo.getpixel((704, 544))
        if(shiny[0]>250 and shiny[1]>230 and shiny[2]<240 and shiny[2]<10):
            print(Fore.YELLOW + 'SHINY!!!!' + Fore.RESET)
            shinies = shinies+1
            device.shell(f'input tap 536 2061')
            time.sleep(0.5)
        else:
            print("Menu button")
            device.shell(f'input tap 927 2021')
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(0.5)
            device.shell(f'input tap 851 1651')
            time.sleep(1.5)
            image = device.screencap()
            with open('./screenshots/screen.png', 'wb') as f:
                f.write(image)
            time.sleep(0.5)
            star = photo.getpixel((232,1427))

            if(star==(255,213,122)):
                print(f"Good IV's")
                time.sleep(1)
                device.shell(f'input tap 851 1651')
                time.sleep(1)
                device.shell(f'input tap 525 2045')

            else:
                print(f"Bad IV's")
                time.sleep(0.6)
                device.shell(f'input tap 851 1651')
                time.sleep(0.6)
                device.shell(f'input tap 927 2021')
                time.sleep(0.6)
                device.shell(f'input tap 935 1848')
                time.sleep(0.6)
                device.shell(f'input tap 540 1250')
                time.sleep(0.5)
                device.shell(f'input tap 525 1170')
                time.sleep(0.5)



    if(pokecaught[0]==176 and pokecaught[1]==234 and pokecaught[2]==197):
        print('Pokemon Caught!')
        caught = caught+1
        ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Shinies: {int(shinies)} | Pokeballs Thrown: {int(thrown)} | XP Gained: {int(xp)} ")
        #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
        device.shell(f'input tap 540 1500')
        time.sleep(2)
        image = device.screencap()
        with open('./screenshots/screen.png', 'wb') as f:
            f.write(image)
        time.sleep(2)
        shiny = photo.getpixel((704, 544))
        if(shiny[0]>250 and shiny[1]>230 and shiny[2]<240 and shiny[2]<10):
            print(Fore.YELLOW + 'SHINY!!!!' + Fore.RESET)
            shinies = shinies+1
            device.shell(f'input tap 536 2061')
            time.sleep(0.5)
        else:
            print("Menu button")
            device.shell(f'input tap 927 2021')
            time.sleep(1)
            device.shell(f'input tap 851 1651')
            time.sleep(0.5)
            device.shell(f'input tap 851 1651')
            time.sleep(1.5)
            image = device.screencap()
            with open('./screenshots/screen.png', 'wb') as f:
                f.write(image)
            time.sleep(0.5)
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
                device.shell(f'input tap 525 1170')
                time.sleep(0.5)
    if(newpokecaught[0]==114 and newpokecaught[1]==214 and newpokecaught[2]==157):
        print('Pokemon Caught! New Pokemon!')
        caught = caught+1
        ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Shinies: {int(shinies)} | Pokeballs Thrown: {int(thrown)} | XP Gained: {int(xp)} ")
        #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
        device.shell(f'input tap 540 1500')
        time.sleep(20)
        print("Menu button")
        device.shell(f'input tap 927 2021')
        time.sleep(1)
        device.shell(f'input tap 851 1651')
        time.sleep(1)
        device.shell(f'input tap 851 1651')
        time.sleep(1.5)
        image = device.screencap()
        with open('./screenshots/screen.png', 'wb') as f:
            f.write(image)
        time.sleep(0.5)
        star = photo.getpixel((232,1427))

        if(star[0]>250 and star[1]>205 and star[1]<220 and star[2]>118 and star[2]<128):
            print(f"Good IV's")
            time.sleep(0.7)
            device.shell(f'input tap 851 1651')
            time.sleep(0.7)
            device.shell(f'input tap 525 2045')
        else:
            print(f"Bad IV's")
            time.sleep(0.7)
            device.shell(f'input tap 851 1651')
            time.sleep(0.7)
            device.shell(f'input tap 927 2021')
            time.sleep(0.7)
            device.shell(f'input tap 935 1848')
            time.sleep(0.7)
            device.shell(f'input tap 540 1250')
            time.sleep(0.7)



    if(rando[0]==49 and rando[1]==222 and rando1[2]==255):
        device.shell(f'input touchscreen swipe 200 1000 800 100 100')
        time.sleep(1)
        device.shell(f'input tap 525 2045')
        time.sleep(0.8)
    if(rando1[0]==31 and rando1[1]==143 and rando1[2]==249):
        device.shell(f'input touchscreen swipe 200 1000 800 100 100')
        time.sleep(1)
        device.shell(f'input tap 525 2045')
        time.sleep(0.8)


    if(pokestop[0]>=240 and pokestop[1]>=245 and pokestop[2]>=240):
        print("Pokestop Found!")
        image = device.screencap()

    
    
    
    
        with open('./screenshots/screen.png', 'wb') as f:
            f.write(image)
        fail = photo.getpixel((536, 1643))
        if(fail[0]==28 and fail[1]==100 and fail[2]==203):
            device.shell(f'input tap 538, 2045')
            time.sleep(0.5)

        else:
            pokestops = pokestops+1
            ctypes.windll.kernel32.SetConsoleTitleW(f"PogoBot | Caught: {int(caught)} | Pokestops: {int(pokestops)} | Shinies: {int(shinies)} | Pokeballs Thrown: {int(thrown)} | XP Gained: {int(xp)} ")
            #RPC.update(large_image="pokeball", large_text="Pokemon Bot", details=f"PogoBot\nCaught: {int(caught)}\nPokestops: {int(pokestops)}\nPokeballs Thrown: {int(thrown)}", start=start_time)
            device.shell(f'input touchscreen swipe 200 1000 800 100 100')
            time.sleep(0.2)
            device.shell(f'input tap 525 2045')
            time.sleep(0.4)




    
    if(teamrocket[0]==66 and teamrocket[1]==208 and teamrocket[2]==165):
        print("Team Rocket, Fuck that")
        device.shell(f'input tap 525 2045')
        time.sleep(0.4)

 


