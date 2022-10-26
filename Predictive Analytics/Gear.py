from ast import Mod
import tkinter as tk
import tkinter.font as tkFont
from tkinter import messagebox
from tkinter.messagebox import askokcancel, showinfo, WARNING
from tkinter import *
from PIL import ImageTk, Image  
import datetime

def main(root, Module_val, Pitch_Circle_Diameter_of_Pinion, Pitch_Circle_Diameter_of_Gear, Addendum_val, Deddendum_val):
    
    def getImage(imagepath, w, h) :
        image1 = Image.open(imagepath)
        image1 = image1.resize((w,h))
        test = ImageTk.PhotoImage(image1)
        return test

    #setting title
    root.title("Gear Images")
    #setting window size
    width=750
    height=600
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    root.geometry(alignstr)
    root.resizable(width=False, height=False)

    BGLabel=tk.Label(root)
    BGLabel["bg"] = "#000000"
    BGLabel["justify"] = "center"
    BGLabel.place(x=3,y=2,width=743,height=596)

    Title=tk.Label(root)
    Title["bg"] = "#ffd700"
    ft = tkFont.Font(family='Times',size=33)
    Title["font"] = ft
    Title["fg"] = "#333333"
    Title["justify"] = "center"
    Title["text"] = "Gear Images"
    Title.place(x=20,y=10,width=712,height=85)

    img = getImage("C:/Users/SAMEER/Downloads/spur-gear.jpg", 297, 300)
    Photo1=tk.Button(root,image=img)
    Photo1.image = img
    Photo1["bg"] = "#f6f5f4"
    ft = tkFont.Font(family='Times',size=10)
    Photo1["font"] = ft
    Photo1["fg"] = "#2e3436"
    Photo1["justify"] = "center"
    Photo1["text"] = "Image of room"
    Photo1.place(x=20,y=120,width=350,height=400)

    img = getImage("C:/Users/SAMEER/Downloads/Gear+Terminology.jpg",297,300)
    Photo2=tk.Button(root,image=img)
    Photo2.image = img
    Photo2["bg"] = "#f6f5f4"
    ft = tkFont.Font(family='Times',size=10)
    Photo2["font"] = ft
    Photo2["fg"] = "#2e3436"
    Photo2["justify"] = "center"
    Photo2["text"] = "Image of room"
    Photo2.place(x=380,y=120,width=350,height=400)

    Back=tk.Button(root)
    Back["bg"] = "#ffb800"
    Back["borderwidth"] = "3px"
    ft = tkFont.Font(family='Times',size=23)
    Back["font"] = ft
    Back["fg"] = "#000000"
    Back["justify"] = "center"
    Back["text"] = "Go Back"
    Back.place(x=330,y=535,width=200,height=52)
    Back["command"] = lambda : back(root, Module_val, Pitch_Circle_Diameter_of_Pinion, Pitch_Circle_Diameter_of_Gear, Addendum_val, Deddendum_val)


def back(root, Module_val, Pitch_Circle_Diameter_of_Pinion, Pitch_Circle_Diameter_of_Gear, Addendum_val, Deddendum_val) :
    import Results as output
    output.main(root, Module_val, Pitch_Circle_Diameter_of_Pinion, Pitch_Circle_Diameter_of_Gear, Addendum_val, Deddendum_val)