# -*- coding: utf-8 -*-

#これは、メッセージ購読のプログラムです。

#ライブラリのインポート
import rclpy
from std_msgs.msg import String

#メイン関数
def main(args=None):
    rclpy.init(args=args)
    #ノードの作成
    node = rclpy.create_node('minimal_subscriber')
    #メッセージを購読するトピックを作成
    subscription = node.create_subscription(String, 'topic', lambda msg: node.get_logger().info('I heard: "%s"' % msg.data),10)
    subscription
    #ループに入る
    rclpy.spin(node)
    #プログラムの終了処理
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()