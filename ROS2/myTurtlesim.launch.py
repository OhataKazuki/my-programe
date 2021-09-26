# -*- coding: utf-8 -*-

#このプログラムは、Turtlesimシミュレータを起動するlaunchファイルです。

#ライブラリのインポート
import launch.actions
import launch_ros.actions
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
	return LaunchDescription([
    # Launch時にメッセージを出力する
	launch.actions.LogInfo(
		msg="Launch turtlesim node and turlte_teleop_key node."),
    #Launchしてから3秒後にメッセージを出力する
	launch.actions.TimerAction(period=3.0,actions=[
		launch.actions.LogInfo(
			msg="It's been three minutes since the launch."),
		]),
    #Turtlesim シミュレータの起動
	Node(
		package='turtlesim', #パッケージ名
		node_namespace='turtlesim', #ノードを起動する名前空間
		node_executable='turtlesim_node', #ノードの実行ファイル名
		node_name='turtlesim', #ノード名
		output='screen', #標準出力をコンソールに表示
        #パラメータの値をっ設定する
		parameters=[{'backgrond_r':255},
			{'background_g':255},
			{'background_b':0},]),
    #Turtlesimをキーボードで操作するノードの起動
	Node(
		package='turtlesim', #パッケージ名
		node_namespace='turtlesim', #ノードを起動する名前空間
		node_executable='turtle_teleop_key', #ノードの実行ファイル名
		node_name='teleop_turtle', #ノード名
        #turtle_teleop_keyをxterm上で実行する
		prefix="xterm -e"
	),
])