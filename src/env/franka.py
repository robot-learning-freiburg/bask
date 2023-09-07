import actionlib
import cv2
import numpy as np
import rospy
import tf
import torch
from franka_gripper.msg import GraspAction, GraspGoal
from franka_msgs.msg import ErrorRecoveryActionGoal  # , FrankaState
from geometry_msgs.msg import PoseStamped, Vector3
from loguru import logger
from omegaconf import OmegaConf
from rl_franka import RLControllerManager
from rl_franka.panda import Panda
from rl_franka.panda_controller_manager import PandaControllerManager
from robot_io.cams.threaded_camera import ThreadedCamera
from sensor_msgs.msg import JointState

from config import realsense_conf_path
from env.environment import BaseEnvironment
from utils.geometry import (euler_to_quaternion,
                            homogenous_transform_from_rot_shift,
                            quaternion_to_euler)
from utils.logging import indent_logs
from utils.observation import (CameraOrder, SceneObservation,
                               SingleCamObservation, dict_to_tensordict,
                               empty_batchsize)
from viz.keypoint_selector import COLOR_RED, draw_reticle
from viz.operations import int_to_float_range, np_channel_back2front

realsense_conf = OmegaConf.load(realsense_conf_path)

position_limits = np.array([[-0.6, 0.75],   # x
                            [-0.6, 0.6],    # y
                            [-0.4, 0.9]])   # z


neutral_position = [0.54697693, -0.03563685, 0.36095113]
neutral_quaternion = [0.682471212769279, 0.7300403768682293,
                      0.009188186958328048, 0.034491580186223614]
joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
               'panda_joint5', 'panda_joint6', 'panda_joint7']
neutral_joints = [-0.02233664, 0.05485502, 0.03904168, -1.66654815,
                  -0.01360612, 1.77192928, 0.85765993]


cartesian_controller = 'cartesian_impedance_controller'
joint_controller = 'joint_position_controller'

physical_cameras = {  # Frame name, serial number
    # D435: 934222071497
    'wrist': tuple(('/camera_wrist_depth_optical_frame', 'SERIALNO')),
    'overhead': tuple(('camera_tilt_depth_optical_frame', 'SERIALNO'))
}

dist_coeffs = np.array(
    [-0.0563001148402691, 0.0614541843533516, -0.000408029329264536,
     0.000655462790746242, -0.0205035209655762])


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
    def __init__(self, config):
        super().__init__(config)

        self.camera_names = config["cameras"]

        self.teleop = config["teleop"]
        self.eval = config["eval"]

        logger.info("Initializing ROS...")

        rospy.init_node("franka_teleop_node_gpu")

        self.link_name = 'panda_link0'

        self.robot_pose = PoseStamped()
        self.rot_euler = Vector3()

        self.robot = Panda()
        self.wait_for_initial_pose()

        # self.franka_state_sub = rospy.Subscriber(
        #     "/franka_state_controller/franka_states", FrankaState,
        #     self.update_robot_state)

        if self.teleop:
            logger.info("Setting up teleop...")
            self.pose_publisher = rospy.Publisher(
                "/controllers/cartesian_impedance_controller/equilibrium_pose",
                PoseStamped, queue_size=10)
            self.joint_publisher = rospy.Publisher(
                "/controllers/joint_position_controller/command",
                # "/franka_state_controller/joint_states_desired",
                JointState, queue_size=10)

            rospy.Timer(rospy.Duration(0.005), self.publisher_callback)

            self.grasp_client = actionlib.SimpleActionClient(
                "/franka_gripper/grasp", GraspAction)
            self.grasp_client.wait_for_server()
            self.grasp_state = None

            self.error_recovery_pub = rospy.Publisher(
                "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal,
                queue_size=1)

            self._recover_from_errors()
            self.panda_controller_manager = PandaControllerManager()
            self.panda_controller_manager.set_joint_stiffness_high()
            self.panda_controller_manager.set_cartesian_stiffness_high()

            self.rl_controller_manager = RLControllerManager()
            self.rl_controller_manager.activate_controller(
                cartesian_controller)

        self.joint_state = JointState()
        self.joint_state.name = joint_names
        self.joint_state.position = neutral_joints

        logger.info("Setting up cameras...")

        self.cameras = []
        self.camera_frames = []

        with indent_logs():
            for cam in self.camera_names:
                frame, sn = physical_cameras[cam]

                realsense_conf.serial_number = sn

                self.cameras.append(ThreadedCamera(realsense_conf))
                self.camera_frames.append(frame)

                logger.info("Found cam {} with serial number {}", cam, sn)

        assert len(self.cameras) == len(self.camera_names), \
            "Some camera was not found."

        if not self.cameras:
            print("Found no camera.")

        self.image_size = config["image_crop"]
        self.image_crop = config.get("image_crop", None)

        self.intrinsics = [torch.Tensor(c.get_camera_matrix())
                           for c in self.cameras]

        if (img_crop := self.image_crop) is not None:
            x_offset = img_crop[0]
            y_offset = img_crop[2]

            for i in range(len(self.intrinsics)):
                self.intrinsics[i][0][2] = self.intrinsics[i][0][2] - x_offset
                self.intrinsics[i][1][2] = self.intrinsics[i][1][2] - y_offset

        self.trans = tf.TransformListener()

        self.win_rgb_name = 'Wrist rgb stream'
        self.camera_rgb_window = cv2.namedWindow(self.win_rgb_name, cv2.WINDOW_AUTOSIZE)

        self.reset()

        logger.info("Done!")

    def close(self):
        rospy.signal_shutdown("Keyboard interrupt.")

    def wait_for_initial_pose(self):
        while self.robot.state.O_T_EE is None:
            pass

        O_T_EE = self.robot.state.O_T_EE
        initial_quaternion = tf.transformations.quaternion_from_matrix(O_T_EE)
        initial_quaternion /= np.linalg.norm(initial_quaternion)

        self.robot_pose.pose.orientation.x, \
            self.robot_pose.pose.orientation.y, \
            self.robot_pose.pose.orientation.z, \
            self.robot_pose.pose.orientation.w = initial_quaternion
        self.robot_pose.pose.position.x, \
            self.robot_pose.pose.position.y, \
            self.robot_pose.pose.position.z = O_T_EE.T[3, :3]

        self.rot_euler.x, \
            self.rot_euler.y, \
            self.rot_euler.z = quaternion_to_euler(initial_quaternion)

        self.initial_pos = O_T_EE.T[3, :3]
        self.initial_quaternion = initial_quaternion

        self.initial_q = self.robot.state.q

    @logger.contextualize(filter=False)
    def return_to_neutral_pose(self):
        self._recover_from_errors()

        logger.info("Returning to neutral pose ...")

        self.robot.authorize_reset()
        self.robot.move_joint_position(np.asarray(neutral_joints), 0.15, 0.02)
        self.wait_for_initial_pose()

        with indent_logs():
            logger.info("Done.")

        self.robot.cm.activate_controller(cartesian_controller)

    def _recover_from_errors(self):
        self.robot.cm.recover_error_state()

    def reset(self):
        super().reset()

        if self.teleop:
            self.return_to_neutral_pose()
            self.set_gripper_pose(1)
            self.return_to_neutral_pose()
            # self.set_gripper_pose(1)

        obs = self.get_obs()

        return obs

    def publisher_callback(self, msg):
        self.robot_pose.header.frame_id = self.link_name
        self.robot_pose.header.stamp = rospy.Time(0)

        self.pose_publisher.publish(self.robot_pose)

    # def update_robot_state(self, state):
    #     self.robot_state = state

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

    def _step(self, action: np.ndarray, postprocess: bool = True,
              delay_gripper: bool = True, scale_action: bool = True,
              invert_action : tuple[bool] = tuple((True, False, True)),
              ) -> tuple[SceneObservation, float, bool, dict]:
        """
        Postprocess the action and execute it in the environment.
        Clamps translations to the workspace limits.

        Parameters
        ----------
        action : np.ndarray
            The raw action predicted by a policy.
        postprocess : bool, optional
            Whether to postprocess the action at all, by default True
        delay_gripper : bool, optional
            Whether to delay the gripper action. Usually needed for ML
            policies, by default True
        scale_action : bool, optional
            Whether to scale the action. Usually needed for ML policies,
            by default True
        invert_action: tuple[bool], optional
            Whether to invert the translation in the x, y, z direction.

        Returns
        -------
        SceneObservation, float, bool, dict
            The observation, reward, done flag and info dict.
        """
        if postprocess:
            action = self.postprocess_action(
                action, scale_action=scale_action, delay_gripper=delay_gripper,
                return_euler=True)
        else:
            action = action

        delta_position, delta_rot_euler, gripper = action.split([3, 6])

        for i in range(3):
            if invert_action[i]:
                delta_position[i] = -delta_position[i]

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
        self.rot_euler.x += delta_rot_euler[0]
        self.rot_euler.y += delta_rot_euler[1]
        self.rot_euler.z += delta_rot_euler[2]

        if self.teleop:
            self._set_rotation(euler_to_quaternion(
                [self.rot_euler.x, self.rot_euler.y, self.rot_euler.z]))

            self.set_gripper_pose(gripper)

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
        wrist_position = np.array(self.robot.state.O_T_EE[12:15])
        wrist_quaternion = tf.transformations.quaternion_from_matrix(
            np.transpose(np.reshape(self.robot.state.O_T_EE, (4, 4))))
        wrist_pose = torch.Tensor(
            np.concatenate((wrist_position, wrist_quaternion)))

        joint_pos = torch.Tensor(self.robot.state.q)
        joint_vel = torch.Tensor(self.robot.state.dq)

        gripper_width = torch.Tensor([self.robot.gripper_pos.position])
        print(gripper_width)

        # extrinsics are in homegenous matrix format
        extrinsics = [self.get_camera_pose(f) for f in self.camera_frames]

        cam_img = img_frames[0][:, :, ::-1].copy()  # make contiguous
        # undistorted_image = cv2.undistort(
        #     cam_img, self.intrinsics[0], dist_coeffs)
        # img_frames[0] = undistorted_image[:, :, ::-1].copy()
        undistorted_image = cam_img

        if self.image_crop is not None:
            # NOTE: anything but the left crop line are untested
            image_h, image_w = undistorted_image.shape[:2]
            crop_l, crop_r, crop_t, crop_b = self.image_crop
            display_image = cv2.line(undistorted_image, (crop_l, 0),
                                     (crop_l, image_h), (0, 0, 255), 2)
            display_image = cv2.line(display_image, (image_w - crop_r, 0),
                                     (image_w - crop_r, image_h),
                                     (0, 0, 255), 2)
            display_image = cv2.line(display_image, (0, crop_t),
                                     (image_w, crop_t), (0, 0, 255), 2)
            display_image = cv2.line(display_image, (0, image_h - crop_b),
                                     (image_w, image_h - crop_b),
                                     (0, 0, 255), 2)

        else:
            display_image = undistorted_image

        self._current_cam_img = undistorted_image
        cv2.imshow(self.win_rgb_name, display_image)

        cv2.waitKey(1)

        if self.eval:
            img_frames = [int_to_float_range(np_channel_back2front(f))
                          for f in img_frames]
        else:
            img_frames = [np_channel_back2front(f) for f in img_frames]

        if self.image_crop is not None:
            # NOTE: this is untested!
            l, r, t, b = self.image_crop
            img_frames = [i[:, t:i.shape[-2]-b, l:i.shape[-1]-r]
                          for i in img_frames]
            depth_frames = [i[:, t:i.shape[-2]-b, l:i.shape[-1]-r]
                            for i in depth_frames]

        # NOTE: this is untested.
        camera_obs = {}
        for i, cam in enumerate(self.camera_names):
            camera_obs[cam] = SingleCamObservation(**{
                'rgb': torch.Tensor(img_frames[i]),
                'depth': torch.Tensor(depth_frames[i]),
                'extrinsics': torch.Tensor(extrinsics[i]),
                'intrinsics': self.intrinsics[i],
            }, batch_size=empty_batchsize)

        multicam_obs = dict_to_tensordict(
            {'_order ': CameraOrder._create(self.camera_names)} | camera_obs)

        obs = SceneObservation(cameras=multicam_obs, ee_pose=wrist_pose,
                               joint_pos=joint_pos, joint_vel=joint_vel,
                               gripper_state=gripper_width,
                               batch_size=empty_batchsize)

        return obs

    def update_visualization(self, info: dict) -> None:
        kp_tens = info['vis_encoding'][0]  # First one should be wrist camera

        u, v, _ = kp_tens.chunk(3)
        u = (u/2 + 0.5) * self.image_size[1] + self.image_crop[0]
        v = (v/2 + 0.5) * self.image_size[0] + self.image_crop[2]

        rgb_w_kp = self._current_cam_img.copy()

        for i, j in zip(u, v):
            draw_reticle(rgb_w_kp, int(i), int(j), COLOR_RED)

        cv2.imshow(self.win_rgb_name, rgb_w_kp)

        cv2.waitKey(1)


    def get_camera_pose(self, topic_name):
        cam_position, cam_quaternion = self.trans.lookupTransform('/base_link', topic_name, rospy.Time(0))
        cam_rot_matrix = tf.transformations.quaternion_matrix(cam_quaternion)[:3, :3]
        return homogenous_transform_from_rot_shift(cam_rot_matrix, cam_position)
