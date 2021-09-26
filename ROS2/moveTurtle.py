# -*- coding: utf-8 -*-

#これは、Turtlesimのカメを動かすPyhonプログラムです。


#ライブラリのインポート
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


class MoveTurtle(Node):
	def __init__(self):
        #turtlesim_moveという名のノードを作成します
		super().__init__('turtlesim_move')
        #Twist型のメッセージを配信する/turtle1/cmd_velというトピックを作成します
		self.pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        #定期的に呼ばれるコールバック関数の定義
		self.tmr = self.create_timer(1.0,self.timer_callback)
        #/turtle1/pooseからメッセージを購読する準備
		self.sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)

    #メッセージが更新された時のコールバック関数
	def pose_callback(self, msg):
        #位置と姿勢を画面に表示
		self.get_logger().info('(x,y,theta):[%f%f%f]'%(msg.x,msg.y,msg.theta))
    
    #定期的に呼ばれるコールバック関数
	def timer_callback(self):
		msg = Twist()
        #並進速度のx成分
		msg.linear.x = 1.0
        #回転速度のz成分
		msg.angular.z = 0.5
        #メッセージを配信
		self.pub.publish(msg)

def main(args=None):
	rclpy.init(args=args)
	move=MoveTurtle()
	rclpy.spin(move)


if __name__ == '__main__':
	main()