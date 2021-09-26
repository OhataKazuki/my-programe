# -*- coding: utf-8 -*-

#これは、Turtlesimの亀を生成するプログラミングです。

#ライブラリのインポート
import rclpy
from turtlesim.srv import Spawn

#メイン関数
def main(args=None):
    rclpy.init(args=args)
    #ノードの作成
    node = rclpy.create_node('spawn_client')
    #クライアントノードの作成
    client = node.create_client(Spawn,'/spawn')
    #サービスを呼び出す設定
    req = Spawn.Reqest()
    req.x = 2.0 #x座標
    req.y = 2.0 #y座標
    req.theta = 0.2 #姿勢
    req.name = 'new_turtle' #新しい亀の名前
    #サービスの開始
    while not client.wait_for_service(timeout_sec = 1.0):
        node.get_logger().info('service not available, waiting again...')
    
    #サービスを呼び出す
    future = client.call_async(req)
    rclpy.spin_until_future_comlete(node, future)
    try:
        result = future.result()
    except Exception as e:
        node.get_logger().info('Service call failed %r' %(e,))
    #メッセージを表示
    else:
        node.get_logger().info('Result of x,y,theta, name: %f %f %f %s' %(req.x, req.y, req.theta, result.name))
    
    #プログラムの終了処理
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()