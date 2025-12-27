import rclpy
import math
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point
from std_msgs.msg import Float64, Float64MultiArray, Bool

from sensor_msgs.msg import JointState

def quaternion_from_yaw(yaw: float):
    """Return (x, y, z, w) for a rotation of 'yaw' around z-axis."""
    half = yaw / 2.0
    qx = 0.0
    qy = 0.0
    qz = math.sin(half)
    qw = math.cos(half)
    return qx, qy, qz, qw


class CubeTransformer(Node):
    def __init__(self):
        super().__init__('cube_transformer')
        
        # NEW: store latest joint state so we can command the gripper safely
        self.latest_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/mars/arm/state',
            self.joint_state_callback,
            10,
        )

        # NEW: publisher to the real arm command topic
        self.arm_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/mars/arm/commands',
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.cube_locked = False
        self.latest_cube_base = None

        self.subscription = self.create_subscription(
            PointStamped,
            '/cube_detector/target_point',
            self.cube_callback,
            10,
        )

        self.cube_base_pub = self.create_publisher(
            PointStamped,
            '/cube_detector/target_in_base',
            10,
        )

        self.pre_pub = self.create_publisher(
            PoseStamped,
            '/cube_grasp/pre_grasp',
            10,
        )
        self.grasp_pub = self.create_publisher(
            PoseStamped,
            '/cube_grasp/grasp',
            10,
        )
        self.lift_pub = self.create_publisher(
            PoseStamped,
            '/cube_grasp/lift',
            10,
        )

        self.ik_pub = self.create_publisher(Twist, 'ik_delta', 10)
        self.ik_gripper_pub = self.create_publisher(Bool, 'ik_gripper', 10)
        self.busy = False

        self.latest_cube_base = None
        self.get_logger().info(
            "CubeTransformer node started, listening to /cube_detector/target_point"
        )

        self.motion_active = False                # are we currently driving a target?
        self.motion_start_time = None             # rclpy.Time when we started
        self.motion_duration = 10.0               # seconds to keep publishing
        self.stop_ik_1 = False
        self.stop_open_grip = False
        self.start_ik_2 = False

        # Timer to repeatedly send IK when active
        self.motion_timer = self.create_timer(0.1, self.motion_timer_cb)  # 10 Hz

        self.latest_cube_base = None
        self.get_logger().info(
            "CubeTransformer node started, listening to /cube_detector/target_point"
        )

    def motion_timer_cb(self):
        """Publish the stored IK target for a fixed duration, then stop."""
        if not self.motion_active:
            return

        now = self.get_clock().now()
        elapsed = (now - self.motion_start_time).nanoseconds * 1e-9

        if elapsed > self.motion_duration:
            # Stop after motion_duration seconds
            self.get_logger().info(
                f"Stopping IK commands after {self.motion_duration:.1f} seconds."
            )
            self.stop_ik_1 = True
        if elapsed > (1.5*self.motion_duration):
            self.get_logger().info(
                f"Starting grasp motion after {1.5 * self.motion_duration:.1f} seconds."
            )
            self.stop_open_grip = True
            self.start_ik_2 = True
    
    def joint_state_callback(self, msg: JointState):
        """Store the latest arm joint positions from /mars/arm/state."""
        self.latest_joint_state = msg

    def cube_callback(self, msg: PointStamped):

        self.get_logger().info(
            f"Cube in camera frame {msg.header.frame_id}: "
            f"x={msg.point.x:.3f}, y={msg.point.y:.3f}, z={msg.point.z:.3f}"
        )

        try:
            target_frame = 'base_link'

            # Use latest TF to avoid extrapolation issues
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                msg.header.frame_id,
                Time(),  # latest transform
                timeout=Duration(seconds=0.2),
            )

            # Transform into base frame
            cube_base = do_transform_point(msg, transform)

            if not self.cube_locked:
                self.latest_cube_base = cube_base
            self.cube_base_pub.publish(self.latest_cube_base)

            self.get_logger().info(
                f"Cube in BASE frame {target_frame}: "
                f"x={cube_base.point.x:.3f}, y={cube_base.point.y:.3f}, z={cube_base.point.z:.3f}"
            )

            # Compute poses first
            pre, grasp, lift = self.make_grasp_poses(self.latest_cube_base)
            if pre is None:
                return

            # Now publish them
            self.pre_pub.publish(pre)
            self.grasp_pub.publish(grasp)
            self.lift_pub.publish(lift)

            # Log for debugging
            self.get_logger().info(
                f"Pre-grasp:  x={pre.pose.position.x:.3f}, "
                f"y={pre.pose.position.y:.3f}, z={pre.pose.position.z:.3f}"
            )
            self.get_logger().info(
                f"Grasp:      x={grasp.pose.position.x:.3f}, "
                f"y={grasp.pose.position.y:.3f}, z={grasp.pose.position.z:.3f}"
            )
            self.get_logger().info(
                f"Lift:       x={lift.pose.position.x:.3f}, "
                f"y={lift.pose.position.y:.3f}, z={lift.pose.position.z:.3f}"
            )

            # === Use PRE-GRASP pose as IK target ===
            target = pre

            # Position in base_link
            x = target.pose.position.x
            y = target.pose.position.y
            z = target.pose.position.z

            # Recover yaw from quaternion (we created it as pure yaw)
            qw = target.pose.orientation.w
            qz = target.pose.orientation.z
            yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * (qz ** 2))

            # Build Twist for IK (interpreted as ABSOLUTE pose in base_link)
            ik_twist = Twist()
            ik_twist.linear.x  = x + 0.01
            ik_twist.linear.y  = y
            ik_twist.linear.z  = z + 0.23

            ik_twist.angular.x = 0.0   # roll
            ik_twist.angular.y = 0.0   # pitch
            ik_twist.angular.z = qz   # yaw

            self.get_logger().info(
                f"Sending IK target: pos=({x:.3f}, {y:.3f}, {z:.3f}), yaw={yaw:.3f}"
            )
            if not self.stop_ik_1:
                gripper_msg = Bool()
                gripper_msg.data = True
                self.ik_gripper_pub.publish(gripper_msg)
                self.ik_pub.publish(ik_twist)

            if self.latest_joint_state is None:
                return
            
            positions = list(self.latest_joint_state.position)

            if len(positions) < 6:
                self.get_logger().warn(
                    f"Expected at least 6 joints in /mars/arm/state, got {len(positions)}"
                )
                return
            
            positions[5] = 1.0

            cmd = Float64MultiArray()
            cmd.data = positions
            if self.stop_ik_1 and not self.stop_open_grip:
                self.arm_cmd_pub.publish(cmd)
                return

            
            if not self.motion_active:
                self.motion_start_time = self.get_clock().now()
            self.motion_active = True

            target = grasp

            # Position in base_link
            x = target.pose.position.x 
            y = target.pose.position.y
            z = target.pose.position.z

            # Recover yaw from quaternion (we created it as pure yaw)
            qw = target.pose.orientation.w
            qz = target.pose.orientation.z
            yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * (qz ** 2))

            # Build Twist for IK (interpreted as ABSOLUTE pose in base_link)
            ik_twist = Twist()
            ik_twist.linear.x  = x - 0.055
            ik_twist.linear.y  = y + 0.01
            ik_twist.linear.z  = z + 0.2

            ik_twist.angular.x = 0.0   # roll
            ik_twist.angular.y = 0.0   # pitch
            ik_twist.angular.z = qz   # yaw

            if self.start_ik_2:
                gripper_msg = Bool()
                gripper_msg.data = False
                self.ik_gripper_pub.publish(gripper_msg)
                self.ik_pub.publish(ik_twist)

            # For now, just send once
            self.busy = True
            self.cube_locked=True
        except TransformException as ex:
            self.get_logger().warn(f"TF transform failed: {ex}")

    def make_grasp_poses(self, cube_base: PointStamped):
        x_c = cube_base.point.x
        y_c = cube_base.point.y
        z_c = cube_base.point.z

        frame = 'base_link'

        r = math.sqrt(x_c**2 + y_c**2)
        if r < 1e-4:
            self.get_logger().warn(
                "Cube too close to base origin; cannot define approach direction"
            )
            return None, None, None

        self.get_logger().info(f"Cube distance in base frame r = {r:.3f} m")

        ux = x_c / r
        uy = y_c / r

        approach_distance = 0.15   # m behind cube for pre-grasp
        contact_offset    = 0.02   # m behind cube at grasp

        z_grasp = z_c # was z_c but changed to fixed value per Chris' request
        yaw = math.atan2(y_c, x_c)
        qx, qy, qz, qw = quaternion_from_yaw(yaw)

        # PRE-GRASP
        pre = PoseStamped()
        pre.header.frame_id = frame
        pre.header.stamp = self.get_clock().now().to_msg()
        pre.pose.position.x = x_c - approach_distance * ux
        pre.pose.position.y = y_c - approach_distance * uy
        pre.pose.position.z = z_grasp
        pre.pose.orientation.x = qx
        pre.pose.orientation.y = qy
        pre.pose.orientation.z = qz
        pre.pose.orientation.w = qw

        # GRASP
        grasp = PoseStamped()
        grasp.header.frame_id = frame
        grasp.header.stamp = self.get_clock().now().to_msg()
        grasp.pose.position.x = x_c - contact_offset * ux
        grasp.pose.position.y = y_c - contact_offset * uy
        grasp.pose.position.z = z_grasp
        grasp.pose.orientation.x = qx
        grasp.pose.orientation.y = qy
        grasp.pose.orientation.z = qz
        grasp.pose.orientation.w = qw

        # LIFT
        lift = PoseStamped()
        lift.header.frame_id = frame
        lift.header.stamp = self.get_clock().now().to_msg()
        lift.pose.position.x = grasp.pose.position.x
        lift.pose.position.y = grasp.pose.position.y
        lift.pose.position.z = z_grasp + 0.15
        lift.pose.orientation.x = qx
        lift.pose.orientation.y = qy
        lift.pose.orientation.z = qz
        lift.pose.orientation.w = qw

        return pre, grasp, lift


def main(args=None):
    rclpy.init(args=args)
    node = CubeTransformer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
