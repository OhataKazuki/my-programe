# -*- coding: utf-8 -*-

#これは、メッセージ配信のプログラムです。


#ライブラリのインポート
import rclpy
from std_msgs.msg import String


def main(args=None):
    rclpy.init(args=args)
    #ノードの作成
    node = rclpy.create_node('minimal_publisher')
    #String型のメッセージを配信するトピックを作成
    publisher = node.create_publisher(String, 'topic', 10)
    msg = String()
    i = 0

    #コールバック関数の定義
    def timer_callback():
        nonlocal i
        msg.data = 'Hello World: %d' %i
        i += 1
        node.get_logger().info('Publishing: "%s"' %msg.data)
        publisher.publish(msg)
        
    #0.5秒に1回配信する
    timer_piriod = 0.5
    timer = node.create_timer(timer_piriod, timer_callback)
    #ループに入る
    rclpy.spin(node)
    #プログラムの終了処理
    node.destroy_timer(timer)
    node.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    main()