"""
RGBの数字で計算をすることで絵具を混ぜるような体験が出来るプログラム
"""

import tkinter
import random

def rgb2hex(r,g,b): #RGB配列からHTMLなどに使われる16進数表現へ
    # r , g , b = 0 〜 255 # int型
    color = (r, g , b)
    html_color = '#%02X%02X%02X' % (color[0],color[1],color[2])
    return html_color

def click_btn():
    red = 0
    green = 0
    blue = 0
    for i in range(7):
        if bvar[i].get() == True:
            red += colors[i][0]
            green += colors[i][1]
            blue += colors[i][2]
    red = int(red  % 256) 
    green = int(green  % 256) 
    blue = int(blue % 256) 
    print((red,green,blue))
    text.delete("1.0", tkinter.END)
    text.insert("1.0", "混ぜたらこんな色になりました。")
    label = tkinter.Label(root,width = 40, height =3, bg =rgb2hex(red, green, blue) )
    label.place(x = 320,y = 70)

root = tkinter.Tk()
root.title("絵具混ぜ混ぜアプリ")
root.resizable(False, False)
canvas = tkinter.Canvas(root, width=800, height=600)
canvas.pack()
gazou = tkinter.PhotoImage(file="haikei.png")
canvas.create_image(400, 300, image=gazou)
button = tkinter.Button(text="MIX!!", font=("Times New Roman", 32), bg="lightgreen", command=click_btn)
button.place(x=400, y=480)
text = tkinter.Text(width=40, height=1, font=("Times New Roman", 16))
text.place(x=320, y=30)

bvar = [None]*7
cbtn = [None]*7
ITEM = [
"赤",
"ゴールド",
"ミディアムストレートブルー",
"ダークグリーン",
"ドジャーブルー",
"シエナ",
"ダークグレー"
]

colors = [
[255,0,0],
[255,215,0],
[123,104,238],
[0,100,0],
[30,144,255],
[160,82,45],
[169,169,169] 
]  
    
for i in range(7):
    bvar[i] = tkinter.BooleanVar()
    bvar[i].set(False)
    cbtn[i] = tkinter.Checkbutton(text=ITEM[i], font=("Times New Roman", 12), variable=bvar[i], bg=rgb2hex(colors[i][0], colors[i][1], colors[i][2]))
    cbtn[i].place(x=400, y=160+40*i)
root.mainloop()
