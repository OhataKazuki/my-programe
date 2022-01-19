"""
画像によるおみくじを表示させるプログラム
"""

import tkinter
import random

def click_btn():
    #サブ画面
    label.destroy()
    message = random.choice([daikiti, tyuukiti, shoukiti, suekiti, daikyou])
    label_2 = tkinter.Label(root,image = message)
    label_2.place(x = 450,y = 30)

root = tkinter.Tk()
root.title("おみくじソフト")
root.resizable(False, False)
canvas = tkinter.Canvas(root, width=800, height=600)
canvas.pack()
gazou = tkinter.PhotoImage(file="jinja.png")
daikiti = tkinter.PhotoImage(file = "大吉.png")
tyuukiti = tkinter.PhotoImage(file = "中吉.png")
shoukiti = tkinter.PhotoImage(file = "小吉.png")
suekiti = tkinter.PhotoImage(file = "末吉.png")
daikyou = tkinter.PhotoImage(file = "大凶.png")
canvas.create_image(400, 300, image=gazou)
label = tkinter.Label(root, text="？？", font=("Times New Roman", 120), bg="white")
label.place(x=380, y=60)
button = tkinter.Button(root, text="おみくじを引く", font=("Times New Roman", 36), command=click_btn, fg="skyblue")
button.place(x=360, y=500)

root.mainloop()
