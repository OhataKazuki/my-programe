"""
バトルの展開を自動で行い、運用素を追加した改造プログラム
"""

import tkinter
import time
import random
FNT = ("Times New Roman", 24)

class GameCharacter:
    def __init__(self, name, life, x, y, imgfile, tagname):
        self.name = name
        self.life = life
        self.lmax = life
        self.x = x
        self.y = y
        self.img = tkinter.PhotoImage(file=imgfile)
        self.tagname = tagname

    def draw(self):
        x = self.x
        y = self.y
        canvas.create_image(x, y, image=self.img, tag=self.tagname)
        canvas.create_text(x, y+120, text=self.name, font=FNT, fill="red", tag=self.tagname)
        canvas.create_text(x, y+200, text="life{}/{}".format(self.life, self.lmax), font=FNT, fill="lime", tag=self.tagname)

    def attack(self):
        dir = 1
        if self.x >= 400:
            dir = -1
        for i in range(5): # 攻撃動作（横に動かす）
            canvas.coords(self.tagname, self.x+i*10*dir, self.y)
            canvas.update()
            time.sleep(0.1)
        canvas.coords(self.tagname, self.x, self.y)

    def sordman_damage(self):
        for i in range(5): # ダメージ（画像の点滅）
            self.draw()
            canvas.update()
            time.sleep(0.1)
            canvas.delete(self.tagname)
            canvas.update()
            time.sleep(0.1)
        if random.random() > 0.7:
            self.life = self.life - 20
        else:
            self.life = self.life - 30
        if self.life > 0:
            self.draw()
        else:
            print(self.name+"は倒れた...")

            
    
    def Ninja_damage(self):
        if random.random() > 0.3:
            for i in range(5): # ダメージ（画像の点滅）
                self.draw()
                canvas.update()
                time.sleep(0.1)
                canvas.delete(self.tagname)
                canvas.update()
                time.sleep(0.1)
            self.life = self.life - 30
            if self.life > 0:
                self.draw()
            else:
                print(self.name+"は倒れた...")
        else:
            for i in range(5): # 回避動作（縦に動かす）
                canvas.coords(self.tagname, self.x, self.y-i*10)
                canvas.update()
                time.sleep(0.1)
            canvas.coords(self.tagname, self.x, self.y)

def buttle():
    if random.random() > 0.7:
        character[0].attack()
        character[1].Ninja_damage()
        if character[1].life > 0:
            character[1].attack()
            character[0].sordman_damage()
    else:
        character[1].attack()
        character[0].sordman_damage()
        if character[0].life > 0:
            character[0].attack()
            character[1].Ninja_damage()

root = tkinter.Tk()
root.title("オブジェクト指向でバトル")
canvas = tkinter.Canvas(root, width=800, height=600, bg="white")
canvas.pack()

btn_left = tkinter.Button(text="バトル！！", command=buttle)
btn_left.place(x=350, y=560)

character = [
    GameCharacter("伝説の侍「刹那」", 200, 200, 280, "samurai.png", "LC"),
    GameCharacter("闇の忍者「半蔵」", 160, 600, 280, "ninja.png", "RC")
]
character[0].draw()
character[1].draw()

root.mainloop()
