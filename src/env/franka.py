import os
import time

import actionlib
import cv2
import numpy as np
import rospy
import tf
import torch
from franka_gripper.msg import GraspAction, GraspGoal
from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from geometry_msgs.msg import PoseStamped, Vector3
from loguru import logger
from omegaconf import OmegaConf
from robot_io.cams.threaded_camera import ThreadedCamera

from env.environment import BaseEnvironment
from env.observation import FrankaObservation, FrankaObservationWristOnly
from utils.geometry import (euler_to_quaternion,
                            homogenous_transform_from_rot_shift,
                            quaternion_to_euler)
from viz.keypoint_selector import COLOR_RED, draw_reticle
from viz.operations import int_to_float_range, np_channel_back2front

realsense_conf = OmegaConf.load(
    "/PLEASE_ADAPT_PATH/src/utils/realsense.yaml")

# os.environ["QT_DEBUG_PLUGINS"] = str(1)

position_limits = np.array([[-0.6, 0.75],   # x
                            [-0.6, 0.6],   # y
                            [-0.4, 0.9]])  # z


neutral_position = [0.54697693, -0.03563685, 0.36095113]
neutral_quaternion = [0.01352279, 0.99964728, -0.00304354, 0.022547]

camera_frame_names = ['/camera_wrist_color_optical_frame', '/camera_tilt_color_optical_frame']
camera_serial_numbers = ['ADAPT', 'ME']


def pad_list(source, target_len=2):
    return source + [None] * (target_len - len(source))

def subsample_image(img):
    return torch.nn.functional.interpolate(
            img, size=(256, 256), mode='bilinear', align_corners=True)

def clamp_translation(position, delta, limits):
    goal = position + delta*0.25

    clipped = np.clip(goal, *limits)

    if goal != clipped:
        logger.info("Limiting translation {} to workspace limits.", goal)

    return clipped


class FrankaEnv(BaseEnvironment):
    def __init__(self, config, eval=False):
        super().__init__(config)

        self.wrist_on = True
        self.overhead_on = False

        self.teleop = config["teleop"]

        self.eval = eval

        logger.info("Initializing ROS...")

        rospy.init_node("franka_teleop_node_gpu")

        self.link_name = 'panda_link0'

        self.robot_pose = PoseStamped()
        self.rot_euler = Vector3()

        self.robot_state = self.wait_for_initial_pose()

        self.franka_state_sub = rospy.Subscriber(
            "/franka_state_controller/franka_states", FrankaState,
            self.update_robot_state)

        if self.teleop:
            logger.info("  Setting up teleop...")
            self.pose_publisher = rospy.Publisher(
                "/controllers/cartesian_impedance_controller/equilibrium_pose",
                PoseStamped, queue_size=10)

            rospy.Timer(rospy.Duration(0.005), self.publisher_callback)

            self.grasp_client = actionlib.SimpleActionClient(
                "/franka_gripper/grasp", GraspAction)
            self.grasp_client.wait_for_server()
            self.grasp_state = None

            self.error_recovery_pub = rospy.Publisher(
                "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal,
                queue_size=10)

        logger.info("  Setting up cameras...")

        self.cameras = []
        self.camera_frames = []
        for enabled, frame, sn in zip([self.wrist_on, self.overhead_on], camera_frame_names, camera_serial_numbers):
            if enabled:
                realsense_conf.serial_number = sn
                self.cameras.append(ThreadedCamera(realsense_conf))
                self.camera_frames.append(frame)
                logger.info("    Found cam with serial number {}", sn)

        if not self.cameras:
            print("    Found no camera.")

        self.crop_left = config.get("crop_left", None)

        self.intrinsics = [c.get_camera_matrix() for c in self.cameras]

        if (x_offset := self.crop_left) is not None:
            for i in range(len(self.intrinsics)):
                self.intrinsics[i][0][2] = self.intrinsics[i][0][2] - x_offset

        self.trans = tf.TransformListener()

        self.win_rgb_name = 'Wrist rgb stream'
        self.camera_rgb_window = cv2.namedWindow(self.win_rgb_name, cv2.WINDOW_AUTOSIZE)

        self.reset()

        logger.info("  Done!")

    def update_cam_viz(self, img_frames, depth_frames):
        for i, (r, d) in enumerate(zip(img_frames, depth_frames)):
            self.cam_imgs[i][0].set_array(r)
            self.cam_imgs[i][1].set_array(d)

    def shutdown(self):
        rospy.signal_shutdown("Keyboard interrupt.")

    def wait_for_initial_pose(self):
        msg = rospy.wait_for_message("franka_state_controller/franka_states",
                                     FrankaState)
        initial_quaternion = tf.transformations.quaternion_from_matrix(
                np.transpose(np.reshape(msg.O_T_EE, (4, 4))))
        initial_quaternion /= np.linalg.norm(initial_quaternion)

        self.robot_pose.pose.orientation.x = initial_quaternion[0]
        self.robot_pose.pose.orientation.y = initial_quaternion[1]
        self.robot_pose.pose.orientation.z = initial_quaternion[2]
        self.robot_pose.pose.orientation.w = initial_quaternion[3]
        self.robot_pose.pose.position.x = msg.O_T_EE[12]
        self.robot_pose.pose.position.y = msg.O_T_EE[13]
        self.robot_pose.pose.position.z = msg.O_T_EE[14]

        self.rot_euler.x, self.rot_euler.y, self.rot_euler.z = \
            quaternion_to_euler(np.array([
                initial_quaternion[0], initial_quaternion[1],
                initial_quaternion[2], initial_quaternion[3]]))

        self.initial_pos = np.array(msg.O_T_EE[12: 15])
        self.initial_quaternion = initial_quaternion

        return msg

    def return_to_neutral_pose(self):
        logger.info("  Returning to neutral pose ...")
        self._recover_from_errors()
        self._set_position(neutral_position)
        self.rot_euler.x, self.rot_euler.y, self.rot_euler.z = \
            quaternion_to_euler(neutral_quaternion)
        self._set_rotation(neutral_quaternion)
        time.sleep(3)
        logger.info("    Done.")

    def _recover_from_errors(self):
        logger.info("  Recovering from errors ...")
        self.error_recovery_pub.publish(ErrorRecoveryActionGoal())
        logger.info("    Done.")

    def reset(self):
        super().reset()

        if self.teleop:
            self.return_to_neutral_pose()
            self.set_gripper_pose(1)

        obs = self.get_obs()

        return obs

    def publisher_callback(self, msg):
        self.robot_pose.header.frame_id = self.link_name
        self.robot_pose.header.stamp = rospy.Time(0)

        self.pose_publisher.publish(self.robot_pose)

    def update_robot_state(self, state):
        self.robot_state = state

    def set_gripper_pose(self, action):
        width_max = 0.2
        width_min = 0.0
        force = 5  # max: 70N
        speed = 0.1
        epsilon_inner = 0.6
        epsilon_outer = 0.6

        open_grip = action > 0.9
        close_grip = action < 0.1

        if self.grasp_state is None or self.grasp_state != action:
            self.grasp_state = action

            grasp_action = GraspGoal()
            grasp_action.speed = speed
            grasp_action.force = force
            grasp_action.epsilon.inner = epsilon_inner
            grasp_action.epsilon.outer = epsilon_outer

            grasp_action.width = width_max if open_grip else width_min

            self.grasp_client.send_goal(grasp_action)

    def step(self, action, manual_demo=False):
        action_delayed = self.postprocess_action(action,
                                                 manual_demo=manual_demo,
                                                 return_euler=True)

        delta_position, delta_angle_euler, gripper_delayed = \
            action_delayed[:3], action_delayed[3:6], action_delayed[6]

        self.robot_pose.pose.position.x = clamp_translation(
            self.robot_pose.pose.position.x, delta_position[0],
            position_limits[0])

        self.robot_pose.pose.position.y = clamp_translation(
            self.robot_pose.pose.position.y, delta_position[1],
            position_limits[1])

        self.robot_pose.pose.position.z = clamp_translation(
            self.robot_pose.pose.position.z, delta_position[2],
            position_limits[2])

        # delta_angle_euler = quaternion_to_euler(delta_angle_quat)
        self.rot_euler.x += delta_angle_euler[0]
        self.rot_euler.y += delta_angle_euler[1]
        self.rot_euler.z += delta_angle_euler[2]

        if self.teleop:
            self._set_rotation(euler_to_quaternion(
                [self.rot_euler.x, self.rot_euler.y, self.rot_euler.z]))

            self.set_gripper_pose(gripper_delayed)

        obs = self.get_obs()

        reward, done = 1, False

        info = {}

        return obs, reward, done, info

    def _set_position(self, position):
        self.robot_pose.pose.position.x, self.robot_pose.pose.position.y, \
            self.robot_pose.pose.position.z = position

    def _set_rotation(self, quaternion):
        self.robot_pose.pose.orientation.x, \
            self.robot_pose.pose.orientation.y, \
            self.robot_pose.pose.orientation.z, \
            self.robot_pose.pose.orientation.w = quaternion

    def get_obs(self):
        frames = [cam.get_image() for cam in self.cameras]
        img_frames = [f[0] for f in frames]
        depth_frames = [f[1] for f in frames]

        # gripper and wrist pose are coordinates + quaternions in world frame
        wrist_position = np.array(self.robot_state.O_T_EE[12:15])
        wrist_quaternion = tf.transformations.quaternion_from_matrix(
                                np.transpose(np.reshape(self.robot_state.O_T_EE, (4, 4))))
        wrist_pose = np.concatenate((wrist_position, wrist_quaternion))

        gripper_pose = wrist_pose

        proprio_obs = self.robot_state.q

        # extrinsics are in homegenous matrix format
        extrinsics = [self.get_camera_pose(f) for f in self.camera_frames]

        cam_img = img_frames[0][:, :, ::-1].copy()  # make contiguous
        cam_img = cv2.line(cam_img, (self.crop_left, 0),
                           (self.crop_left, 480), (0, 0, 255), 2)
        self._current_cam_img = cam_img
        cv2.imshow(self.win_rgb_name, cam_img)

        cv2.waitKey(1)

        if self.eval:
            img_frames = [int_to_float_range(np_channel_back2front(f))
                          for f in img_frames]
        else:
            img_frames = [np_channel_back2front(f) for f in img_frames]

        if self.crop_left is not None:
            img_frames = [i[:, :, self.crop_left:] for i in img_frames]
            depth_frames = [i[:, self.crop_left:] for i in depth_frames]

        Observation = FrankaObservationWristOnly if not self.overhead_on else FrankaObservation

        return Observation(gripper_pose, proprio_obs, wrist_pose, *img_frames,
                           *depth_frames, *extrinsics, *self.intrinsics)

    def update_viz(self, cam_rgb, kp_tens):
        u, v, _ = kp_tens.chunk(3)
        u = (u/2 + 0.5) *480 + self.crop_left
        v = (v/2 + 0.5) * 480
        rgb_w_kp = self._current_cam_img.copy()
        for i, j in zip(u, v):
            draw_reticle(rgb_w_kp, int(i), int(j), COLOR_RED)

        cv2.imshow(self.win_rgb_name, rgb_w_kp)

        cv2.waitKey(1)


    def get_camera_pose(self, topic_name):
        cam_position, cam_quaternion = self.trans.lookupTransform('/base_link', topic_name, rospy.Time(0))
        cam_rot_matrix = tf.transformations.quaternion_matrix(cam_quaternion)[:3, :3]
        return homogenous_transform_from_rot_shift(cam_rot_matrix, cam_position)
