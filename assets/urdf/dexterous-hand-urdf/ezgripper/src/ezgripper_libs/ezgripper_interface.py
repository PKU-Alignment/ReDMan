#!/usr/bin/python

#####################################################################
# Software License Agreement (BSD License)
#
# Copyright (c) 2015, SAKE Robotics
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
##

import rospy
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from std_srvs.srv import Empty

# http://docs.ros.org/indigo/api/control_msgs/html/msg/GripperCommand.html
# float64 position  # if 0, torque mode, if >0 to 100 correlates to 0-100% rotation range
# float64 max_effort  # if 0, torque released,  if >0 to 100 increasing torque
#
# http://docs.ros.org/indigo/api/control_msgs/html/action/GripperCommand.html
# GripperCommand command
# ---
# float64 position  # The current gripper gap size (% rotation of EZGripper fingers)  (NOT in meters)
# float64 effort    # The current effort exerted (% available, NOT in Newtons)
# bool stalled      # True iff the gripper is exerting max effort and not moving
# bool reached_goal # True iff the gripper position has reached the commanded setpoint
#


class EZGripper(object):
    def __init__(self, name):
        self.name = name
        self._grip_max = 100.0 #maximum open position for grippers - correlates to .17 meters
        self._grip_value = self._grip_max
        self._grip_min = 0.01 #if 0.0, torque mode, not position mode
        self._grip_step = self._grip_max/15 # gripper step Cross Up and Cross Down
        self._connect_to_gripper_action()
        self._connect_to_calibrate_srv()

    def _connect_to_gripper_action(self):
        rospy.loginfo("Waiting for action server %s..."%self.name)
        self._client = actionlib.SimpleActionClient(self.name, GripperCommandAction)
        self._client.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Connected to action server")

    def _connect_to_calibrate_srv(self):
        service_name = self.name + '/calibrate'
        rospy.loginfo("Waiting for service %s..."%service_name)
        rospy.wait_for_service(service_name)
        self._calibrate_srv = rospy.ServiceProxy(service_name, Empty)
        rospy.loginfo("Connected to service " + service_name)

    def calibrate(self):
        rospy.loginfo("ezgripper_interface: calibrate")
        try:
            self._calibrate_srv()
        except rospy.ServiceException as exc:
            rospy.logwarn("Service did not process request: " + str(exc))
        else:
            self._grip_value = self._grip_max
        rospy.loginfo("ezgripper_interface: calibrate done")

    def open_step(self):
        self._grip_value = self._grip_value + self._grip_step
        if self._grip_value > self._grip_max:
            self._grip_value = self._grip_max
        rospy.loginfo("ezgripper_interface: goto position %.3f"%self._grip_value)
        goal = GripperCommandGoal()
        goal.command.position = self._grip_value
        goal.command.max_effort = 50.0
        self._client.send_goal_and_wait(goal)
        rospy.loginfo("ezgripper_interface: goto position done")

    def close_step(self):
        self._grip_value = self._grip_value - self._grip_step
        if self._grip_value < self._grip_min:
            self._grip_value = self._grip_min
        rospy.loginfo("ezgripper_interface: goto position %.3f"%self._grip_value)
        goal = GripperCommandGoal()
        goal.command.position = self._grip_value
        goal.command.max_effort = 50.0
        self._client.send_goal_and_wait(goal)
        rospy.loginfo("ezgripper_interface: goto position done")

    def close(self, max_effort): 
        rospy.loginfo("ezgripper_interface: close, effort %.1f"%max_effort)
        goal = GripperCommandGoal()
        goal.command.position = 0.0
        goal.command.max_effort = max_effort
        self._client.send_goal_and_wait(goal)
        rospy.loginfo("ezgripper_interface: close done")
        self._grip_value = self._grip_min

    def hard_close(self):
        rospy.loginfo("ezgripper_interface: hard close")
        goal = GripperCommandGoal()
        goal.command.position = 0.0
        goal.command.max_effort = 100 # >0 to 100
        self._client.send_goal_and_wait(goal)
        rospy.loginfo("ezgripper_interface: hard close done")
        self._grip_value = self._grip_min

    def soft_close(self):
        rospy.loginfo("ezgripper_interface: soft close")
        goal = GripperCommandGoal()
        goal.command.position = 0.0
        goal.command.max_effort = 20.0 # >0 to 100
        self._client.send_goal_and_wait(goal)
        rospy.loginfo("ezgripper_interface: soft close done")
        self._grip_value = self._grip_min

    def open(self):
        rospy.loginfo("ezgripper_interface: open")
        goal = GripperCommandGoal()
        goal.command.position = 100.0   # 100% range (0..100)
        goal.command.max_effort = 100.0 # >0 to 100
        self._client.send_goal_and_wait(goal)
        rospy.loginfo("ezgripper_interface: open done")
        self._grip_value = self._grip_max

    def goto_position(self, grip_position = 5.0, grip_effort = 20.0):
        # position in % 0 to 100 (0 is closed), effort in % 0 to 100
        rospy.loginfo("ezgripper_interface: goto position %.3f" %grip_position)
        goal = GripperCommandGoal()
        goal.command.position = grip_position   # range(0.01 to 1.0)
        goal.command.max_effort = grip_effort #  >0 to 100, if 0.0, torque is released
        self._client.send_goal_and_wait(goal)
        rospy.loginfo("ezgripper_interface: goto position done")
        self._grip_value = grip_position

    def release(self):
        rospy.loginfo("ezgripper_interface: release")
        goal = GripperCommandGoal()
        goal.command.position = 0.0 # not dependent on position
        goal.command.max_effort = 0.0 # max_effort = 0.0 releases all torque on motor
        self._client.send_goal_and_wait(goal)
        rospy.loginfo("ezgripper_interface: release done")
        self._grip_value = self._grip_min

if __name__ == "__main__":
    rospy.init_node("ezgripper_client")
    ez = EZGripper('ezgripper/main')
    ez.open()
    ez.calibrate()
    ez.open()
    ez.hard_close()
    ez.open()
    ez.soft_close()
    ez.open()
    rospy.loginfo("Exiting")
