"""
このプログラムは、任意の大きさの迷路を自動生成するプログラムです。
しかし一筆書きできるかは、運しだい...
"""

#ライブラリのインポート
import tkinter
import tkinter.messagebox
import random

#キーに関する設定
key = ""
def key_down(e):
    global key
    key = e.keysym
def key_up(e):
    global key
    key = ""

#変数の定義
mx = 1 #x座標
my = 1 #y座標
yuka = 0 #塗る床の面積

#パズルの大きさ定義
yoko = 9 
tate = 6

#迷路作成用変数
width = yoko + 2
height = tate + 2
block_num = 5 #ブロックの個数 
block_position = [] #ブロックの場所
initialazed = 0 #初期化フラグ

#迷路定義関数
def maze_definition():
    
    global initialazed,block_position
    
    #迷路を定義
    maze = []
    for i in range(tate):
        #移動箇所作成
        maze.append([0] * width)
        #左右の壁を追加
        maze[i][0] = 1
        maze[i][-1] = 1
    #上下の壁を追加
    maze.insert(0,[1] * width)
    maze.append([1] * width)
    
    #ブロックの場所を決める
    if initialazed == 0:
        block_position = set_block()
        initialazed = 1
    
    #ブロックの場所にブロックを設置
    for i in block_position:
        maze[i[0]][i[1]] = 1
    
    return maze

#ブロックの場所をランダムで設定
def set_block():
    
    _block_position = []
    count = 0
    #block_num分追加
    while count <= block_num:
        position_1 = random.randint(2, tate-1)
        position_2 = random.randint(2, yoko-1)
        #ブロックが（1,1）以外、新しい座標の時に追加
        if (position_1,position_2) != (1,1):
            if not (position_1,position_2) in block_position:
                _block_position.append((position_1,position_2))
                count += 1
  
    return _block_position

#操作のメイン関数
def main_proc():
    global mx, my, yuka
    #リセット
    if key == "Shift_L" and yuka > 1:
        canvas.delete("PAINT")
        mx = 1
        my = 1
        yuka = 0
        for y in range(height):
            for x in range(width):
                if maze[y][x] == 2:
                    maze[y][x] = 0
    #キャラの動作
    if key == "Up" and maze[my-1][mx] == 0:
        my = my - 1
    if key == "Down" and maze[my+1][mx] == 0:
        my = my + 1
    if key == "Left" and maze[my][mx-1] == 0:
        mx = mx - 1
    if key == "Right" and maze[my][mx+1] == 0:
        mx = mx + 1
    #床のペイント
    if maze[my][mx] == 0:
        maze[my][mx] = 2
        yuka = yuka + 1
        canvas.create_rectangle(mx*80, my*80, mx*80+79, my*80+79, fill="pink", width=0, tag="PAINT")
    #キャラの移動
    canvas.delete("MYCHR")
    canvas.create_image(mx*80+40, my*80+40, image=img, tag="MYCHR")
    #終了判定
    if yuka == (yoko * tate) - block_num:
        canvas.update()
        tkinter.messagebox.showinfo("おめでとう！", "全ての床を塗りました！")
    else:
        root.after(300, main_proc) #繰り返す

#ウインドウのインスタンス作成
root = tkinter.Tk()
root.title("迷路を塗るにゃん")
root.bind("<KeyPress>", key_down)
root.bind("<KeyRelease>", key_up)
#表示画面の設定
canvas = tkinter.Canvas(width=80 * width, height=80 * height, bg="white")
canvas.pack()

#迷路の作成
maze = maze_definition()

"""
オリジナルの迷路
maze = [
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,1],
    [1,0,1,1,0,0,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,1],
    [1,0,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1]
    ]
"""

#迷路の描画
for y in range(height):
    for x in range(width):
        if maze[y][x] == 1:
            canvas.create_rectangle(x*80, y*80, x*80+79, y*80+79, fill="skyblue", width=0)

#画像の表示
img = tkinter.PhotoImage(file="mimi_s.png")
canvas.create_image(mx*80+40, my*80+40, image=img, tag="MYCHR")

#プログラムの実行
main_proc()
root.mainloop()
