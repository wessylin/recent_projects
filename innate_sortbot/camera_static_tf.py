import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np

class CameraToBaseStaticTF(Node):
    def __init__(self):
        super().__init__('camera_to_base_static_tf')
        self.br = StaticTransformBroadcaster(self)

        # G_base->camera (homogeneous transform)
        G = np.array([
            [ 0.000, -0.174,  0.985, 0.010],
            [-1.000,  0.000,  0.000, 0.007],
            [-0.000, -0.985, -0.174, 0.198],
            [ 0.000,  0.000,  0.000, 1.000],
        ])

        M = G[:3, :3]
        t = G[:3, 3]

        tf_msg = TransformStamped()
        tf_msg.header.frame_id = "base_link"                   # parent
        tf_msg.child_frame_id  = "oak_rgb_camera_optical_frame"  # child

        tf_msg.transform.translation.x = float(t[0])
        tf_msg.transform.translation.y = float(t[1])
        tf_msg.transform.translation.z = float(t[2])

        q = R.from_matrix(M).as_quat()  # (x, y, z, w)
        tf_msg.transform.rotation.x = float(q[0])
        tf_msg.transform.rotation.y = float(q[1])
        tf_msg.transform.rotation.z = float(q[2])
        tf_msg.transform.rotation.w = float(q[3])

        self.transform = tf_msg
        self.timer = self.create_timer(0.1, self.broadcast_tf)

        self.get_logger().info(
            f"Publishing static TF: {tf_msg.header.frame_id} -> {tf_msg.child_frame_id}"
        )

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)

def main(args=None):
    rclpy.init(args=args)
    node = CameraToBaseStaticTF()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
