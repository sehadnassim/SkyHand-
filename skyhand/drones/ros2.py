"""ROS 2 backend: publishes geometry_msgs/Twist (body frame, ROS FLU) and optionally reads nav_msgs/Odometry.

Use this to plug into an existing DJI velocity-control node (PSDK/OSDK ROS wrapper, etc.).
Takeoff/land are sent as std_msgs/String on `cmd_topic` ("takeoff"/"land"); remap to your node's service if needed.
"""
import math
import threading

from ..mapping import VelCmd
from . import Drone


class Ros2Drone(Drone):
    name = "ros2"

    def __init__(self, vel_topic="/drone/cmd_vel", cmd_topic="/drone/command", odom_topic="", **_):
        import rclpy
        from geometry_msgs.msg import Twist
        from std_msgs.msg import String

        rclpy.init()
        self._Twist, self._String = Twist, String
        self.node = rclpy.create_node("skyhand_teleop")
        self.vel_pub = self.node.create_publisher(Twist, vel_topic, 10)
        self.cmd_pub = self.node.create_publisher(String, cmd_topic, 10)
        self._state = None
        self._origin = None
        if odom_topic:
            from nav_msgs.msg import Odometry
            self.node.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def _on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw_ccw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if self._origin is None:
            self._origin = (p.x, p.y, p.z, yaw_ccw)
        ox, oy, oz, oyaw = self._origin
        dx, dy = p.x - ox, p.y - oy
        c, s = math.cos(-oyaw), math.sin(-oyaw)
        fx, fl = c * dx - s * dy, s * dx + c * dy  # in takeoff-heading frame (FLU)
        self._state = {"x": fx, "y": -fl, "z": p.z - oz, "yaw": -(yaw_ccw - oyaw), "airborne": p.z - oz > 0.2}

    def send_velocity(self, cmd: VelCmd):
        t = self._Twist()
        t.linear.x, t.linear.y, t.linear.z = cmd.vx, -cmd.vy, cmd.vz
        t.angular.z = -cmd.yaw_rate
        self.vel_pub.publish(t)

    def _send(self, s):
        m = self._String()
        m.data = s
        self.cmd_pub.publish(m)

    def takeoff(self):
        self._send("takeoff")

    def land(self):
        self._send("land")

    def state(self):
        return self._state

    def close(self):
        self.send_velocity(VelCmd())
        self.node.destroy_node()
