"""
このプログラムは、Tkinterを利用したおみくじのコードです。
フリーの画像素材を使用して文字だけでなく画像の表示方法を学ぶことが目的です。
"""

#ライブラリのインポート
import tkinter
import random

#改造ポイント---------------------------------------
# ボタンが押された時のイベント処理
def click_btn():
    #ヴィジェットの差し替え処理
    label.destroy()
    message = random.choice([daikiti, tyuukiti, shoukiti, suekiti, daikyou])
    label_2 = tkinter.Label(root,image = message)
    label_2.place(x = 450,y = 30)
#------------------------------------------------


#メインウインドウのインスタンス作成
root = tkinter.Tk()
root.title("おみくじソフト")
root.resizable(False, False)
#表示画面の定義
canvas = tkinter.Canvas(root, width=800, height=600)
#表示画面の表示
canvas.pack()

#改造ポイント--------------------------------------------------
#画像のオブジェクト定義
gazou = tkinter.PhotoImage(file="jinja.png")
daikiti = tkinter.PhotoImage(file = "大吉.png")
tyuukiti = tkinter.PhotoImage(file = "中吉.png")
shoukiti = tkinter.PhotoImage(file = "小吉.png")
suekiti = tkinter.PhotoImage(file = "末吉.png")
daikyou = tkinter.PhotoImage(file = "大凶.png")
#-----------------------------------------------------------

#画像オブジェクトの表示
canvas.create_image(400, 300, image=gazou)

#ヴィジェットの定義と配置
label = tkinter.Label(root, text="？？", font=("Times New Roman", 120), bg="white")
label.place(x=380, y=60)
button = tkinter.Button(root, text="おみくじを引く", font=("Times New Roman", 36), command=click_btn, fg="skyblue")
button.place(x=360, y=500)

#ループ処理の実行
root.mainloop()
