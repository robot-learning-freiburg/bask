# Based on https://github.com/ivomarvan/samples_and_experiments/blob/master/Multiple_realsense_cameras/multiple_realsense_cameras.py

import sys
from pprint import pprint

import numpy as np
import pyrealsense2 as rs
from loguru import logger
from robot_io.cams.realsense.realsense import Realsense


class RealsenseCamera:
    '''
    Abstraction of any RealsenseCamera
    '''
    __colorizer = rs.colorizer()

    def __init__(
        self,
        serial_number :str,
        name: str
    ):
        self.__serial_number = serial_number
        self.__name = name
        self.__pipeline = None
        self.__started = False
        self.__start_pipeline()

    def __del__(self):
        if self.__started and not self.__pipeline is None:
            self.__pipeline.stop()

    def get_full_name(self):
        return f'{self.__name} ({self.__serial_number})'

    def __start_pipeline(self):
        # Configure depth and color streams
        self.__pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.__serial_number)
        self.__pipeline.start(self.config)
        self.__started = True
        logger.info(f'    {self.get_full_name()} camera is ready.')

    def get_frames(self) -> [rs.frame]:
        '''
        Return a frame do not care about type
        '''
        frameset = self.__pipeline.wait_for_frames()
        if frameset:
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frameset)
            color_frame = aligned_frames.first(rs.stream.color)
            aligned_depth_frame = aligned_frames.get_depth_frame()

            return [color_frame, aligned_depth_frame]
        else:
            return []

    def get_intrinsics(self):
        profile = self.__pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        # print(depth_intrinsics)
        # NOTE: use depth sensor's intrinsics, not the color sensor's as depth dictates
        # the projection in correspondence finding, not the pixel colors.
        # color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        # color_intrinsics = color_profile.get_intrinsics()
        # print(color_intrinsics)

        intrinsics = np.zeros((3, 3))
        intrinsics[0, 0] = depth_intrinsics.fx
        intrinsics[1, 1] = depth_intrinsics.fy
        intrinsics[0, 2] = depth_intrinsics.ppx
        intrinsics[1, 2] = depth_intrinsics.ppy
        intrinsics[2, 2] = 1
        # TODO: skew is 0?

        return intrinsics

    @classmethod
    def get_title(cls, frame: rs.frame, whole: bool) -> str:
        # <pyrealsense2.video_stream_profile: Fisheye(2) 848x800 @ 30fps Y8>
        profile_str = str(frame.profile)
        first_space_pos = profile_str.find(' ')
        whole_title = profile_str[first_space_pos + 1: -1]
        if whole:
            return whole_title
        return whole_title.split(' ')[0]


    @classmethod
    def get_images_from_video_frames(cls, frames: [rs.frame]) -> ([(np.ndarray, rs.frame)] , [rs.frame], int, int):
        '''
        From all the frames, it selects those that can be easily interpreted as pictures.
        Converts them to images and finds the maximum width and maximum height from all of them.
        '''
        max_width = -1
        max_height = -1
        img_frame_tuples = []
        unused_frames = []
        for frame in frames:
            if frame.is_video_frame():
                # if frame.is_depth_frame():
                #     img = np.asanyarray(RealsenseCamera.__colorizer.process(frame).get_data())
                # else:
                #     img = np.asanyarray(frame.get_data())
                #     img = img[...,::-1].copy()  # RGB<->BGR
                img = np.asanyarray(frame.get_data())
                max_height = max(max_height, img.shape[0])
                max_width  = max(max_width, img.shape[1])
                img_frame_tuples.append((img,frame))
            else:
                unused_frames.append(frame)
        return img_frame_tuples, unused_frames, max_width, max_height

    @classmethod
    def get_table_from_text_data_frame(cls, frame: rs.frame, round_ndigits: int = 2, int_len: int = 3) -> (list, rs.frame):
        '''
        Returns list of rows which ase ists of columns.
        Result can be interpreted as table.
        First row is a header.
        @TODO add interpreatation of other than T265 and D415 camera. (I do not have it)
        '''
        title = RealsenseCamera.get_title(frame, whole=False)
        if frame.is_motion_frame():
            motion_data = frame.as_motion_frame().get_motion_data()
            table = [
                ['name', 'x', 'y', 'z'],
                [title, round(motion_data.x, 2), round(motion_data.y, 2), round(motion_data.z, 2)]
            ]
        elif frame.is_pose_frame():
            data = frame.as_pose_frame().get_pose_data()
            table= [
                [title, 'x', 'y', 'z', 'w'],
                ['acceleration', data.acceleration.x, data.acceleration.y, data.acceleration.z, ''],
                ['angular_acceleration', data.angular_acceleration.x, data.angular_acceleration.y,
                 data.angular_acceleration.z, ''],
                ['angular_velocity', data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z, ''],
                ['rotation', data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w],
                ['translation', data.translation.x, data.translation.y, data.translation.z, ''],
                ['velocity', data.velocity.x, data.velocity.y, data.velocity.z, ''],
                ['mapper_confidence', data.mapper_confidence, '', '', ''],
                ['tracker_confidence', data.tracker_confidence, '', '', ''],
            ]
        else:
            sys.stderr.write(f'No frame to date/image convertor for {frame}.\n')
            return [], None
        if not round_ndigits is None:
            # tabled data to formated strings
            for i, row in enumerate(table):
                for j, cell in enumerate(row):
                    if isinstance(cell, float):
                        formated_str = f'{round(cell, round_ndigits):{round_ndigits + 3}.{round_ndigits}}'
                    elif isinstance(cell, int):
                        formated_str = f'{cell:{int_len}}'
                    else:
                        formated_str = str(cell)
                    table[i][j] = formated_str
        return table, frame
