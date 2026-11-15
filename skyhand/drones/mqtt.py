"""MQTT backend: publishes velocity commands to a mosquitto broker the drone is connected to.

  vel_topic  (default drone/cmd_vel):  {"vx": .., "vy": .., "vz": .., "yaw_rate": ..}  (body frame, y right, yaw + = right)
  cmd_topic  (default drone/command):  "takeoff" / "land"  (plain string)
"""
import json

from ..mapping import VelCmd
from . import Drone


class MqttDrone(Drone):
    name = "mqtt"

    def __init__(self, host="192.168.4.3", port=1883, vel_topic="drone/cmd_vel", cmd_topic="drone/command", **_):
        import paho.mqtt.client as mqtt

        self.vel_topic, self.cmd_topic = vel_topic, cmd_topic
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()

    def send_velocity(self, cmd: VelCmd):
        payload = {"vx": cmd.vx, "vy": cmd.vy, "vz": cmd.vz, "yaw_rate": cmd.yaw_rate}
        self.client.publish(self.vel_topic, json.dumps(payload), qos=0)

    def takeoff(self):
        self.client.publish(self.cmd_topic, "takeoff", qos=1)

    def land(self):
        self.client.publish(self.cmd_topic, "land", qos=1)

    def close(self):
        self.send_velocity(VelCmd())
        self.client.loop_stop()
        self.client.disconnect()
