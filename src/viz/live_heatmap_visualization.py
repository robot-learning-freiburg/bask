import random

import cv2
# import matplotlib.pyplot as plt
import numpy as np

import dense_correspondence.correspondence_finder as correspondence_finder
from viz.operations import channel_front2back, scale

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLACK = (0, 0, 0)


WINDOW_NAME = 'correspondence finder'

scale_factor = 2


def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle on the image at the given (u,v) position

    :param img:
    :type img:
    :param u:
    :type u:
    :param v:
    :type v:
    :param label_color:
    :type label_color:
    :return:
    :rtype:
    """
    white = (255, 255, 255)
    cv2.circle(img, (u, v), 4, label_color, 1)
    cv2.circle(img, (u, v), 5, white, 1)
    cv2.circle(img, (u, v), 6, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 2), white, 1)
    cv2.line(img, (u + 1, v), (u + 2, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 2), white, 1)
    cv2.line(img, (u - 1, v), (u - 2, v), white, 1)


def compute_gaussian_kernel_heatmap_from_norm_diffs(norm_diffs, variance):
    """
    Computes and RGB heatmap from norm diffs
    :param norm_diffs: distances in descriptor space to a given keypoint
    :type norm_diffs: numpy array of shape [H,W]
    :param variance: the variance of the kernel
    :type variance:
    :return: RGB image [H,W,3]
    :rtype:
    """

    """
    Computes an RGB heatmap from the norm_diffs
    :param norm_diffs:
    :type norm_diffs:
    :return:
    :rtype:
    """

    heatmap = np.copy(norm_diffs)

    heatmap = np.exp(-heatmap / variance)  # these are now in [0,1]
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color


def display(*rows):
    # n_rows = len(rows)
    # n_cols = max([len(r) for r in rows])
    #
    # matrix = np.zeros((n_rows*img_height, n_cols*img_width, 3))
    #
    # for i, r in enumerate(rows):
    #     for j, c in enumerate(r):
    #         matrix[i*img_height:(i+1)*img_height,
    #                j*img_width:(j+1)*img_width] = c
    #

    matrix = np.vstack([np.hstack(r) for r in rows])

    c, r, _ = matrix.shape

    matrix = cv2.resize(matrix, (r * scale_factor, c * scale_factor))

    cv2.imshow(WINDOW_NAME, matrix)


# def display(window, img):
#     cv2.imshow(window, cv2.resize(
#         img, (img_size * scale_factor, img_size * scale_factor)))


class HeatmapVisualization(object):
    """
    Launches a live interactive heatmap visualization.

    Keypresses:
        n: new set of images
        s: swap images
        p: pause/un-pause
    """

    def __init__(self, replay_memory, encoder, different_objects=False,
                 norm_by_descr_dim=True, cams=["wrist"]):
        self._config = {
            "norm_diff_threshold": 0.05,
            "heatmap_vis_upper_bound": 0.75,
            "blend_weight_original_image": 0.3,
            "randomize_images": False,
            "kernel_variance": 0.25,

            "different_objects": different_objects,
            "contrast_set_fraction": 0.5,
            "norm_by_descr_dim": norm_by_descr_dim
        }
        self.replay_memory = replay_memory
        self.encoder = encoder
        self.cams = cams

        self._paused = False

        self.image_height = replay_memory.scene_data.image_height
        self.image_width = replay_memory.scene_data.image_width

    def _get_new_images(self, cross_scene=True):
        cam_a = random.choice(self.cams)
        cam_b = random.choice(self.cams)
        obs_a, obs_b = \
            self.replay_memory.sample_data_pair(
                cross_scene=cross_scene,
                cam_a=cam_a, cam_b=cam_b,
                contrast_obj=self._config["different_objects"],
                contrast_fraction=self._config["contrast_set_fraction"])

        image_a_rgb = obs_a.cam_rgb
        image_b_rgb = obs_b.cam_rgb

        self.tensor1 = image_a_rgb
        self.tensor2 = image_b_rgb

        self.img1 = cv2.cvtColor(channel_front2back(
            image_a_rgb*255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.img2 = cv2.cvtColor(channel_front2back(
            image_b_rgb*255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _get_new_second_image(self, cross_scene=True):
        cam_b = random.choice(self.cams)
        _, obs_b = \
            self.replay_memory.sample_data_pair(
                cross_scene=cross_scene,
                cam_a=cam_b, cam_b=cam_b,
                contrast_obj=self._config["different_objects"],
                contrast_fraction=self._config["contrast_set_fraction"])

        image_b_rgb = obs_b.cam_rgb

        self.tensor2 = image_b_rgb

        self.img2 = cv2.cvtColor(channel_front2back(
            image_b_rgb*255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _get_descriptor(self, tensor):
        return channel_front2back(self.encoder.compute_descriptor(
                tensor).detach().squeeze(0).cpu()).numpy()

    def _compute_descriptors(self):
        """
        Computes the descriptors for image 1 and image 2 for each network
        :return:
        :rtype:
        """
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY) / 255.0
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY) / 255.0

        self.res1 = self._get_descriptor(self.tensor1)
        self.res2 = self._get_descriptor(self.tensor2)

        self.find_best_match(None, 0, 0, None, None)

    def scale_norm_diffs_to_make_heatmap(self, norm_diffs, threshold):
        """
        TODO (@manuelli) scale with Gaussian kernel instead of linear

        Scales the norm diffs to make a heatmap. This will be scaled between
        0 and 1.
        0 corresponds to a match, 1 to non-match

        :param norm_diffs: The norm diffs
        :type norm_diffs: numpy.array [H,W]
        :return:
        :rtype:
        """

        heatmap = np.copy(norm_diffs)
        greater_than_threshold = np.where(norm_diffs > threshold)
        # linearly scale [0, threshold] to [0, 0.5]
        heatmap = heatmap / threshold * self._config["heatmap_vis_upper_bound"]
        # greater than threshold is set to 1
        heatmap[greater_than_threshold] = 1
        heatmap = heatmap.astype(self.img1_gray.dtype)
        return heatmap

    def find_best_match(self, event, u, v, flags, param):
        """
        For each network, find the best match in the target image to point
        highlighted with reticle in the source image. Displays the result.
        :return:
        :rtype:
        """

        if self._paused:
            return

        u = int(u/scale_factor) % self.image_width
        v = int(v/scale_factor) % self.image_height
        # print("Pos a", u, v)

        img_1_with_reticle = np.copy(self.img1)
        draw_reticle(img_1_with_reticle, u, v, COLOR_GREEN)

        alpha = self._config["blend_weight_original_image"]
        beta = 1 - alpha

        img_2_with_reticle = np.copy(self.img2)

        best_match_uv, best_match_diff, norm_diffs_2 = \
            correspondence_finder.find_best_match((u, v), self.res1, self.res2)

        D = self.res1.shape[-1]

        norm_diffs_1 = correspondence_finder.get_norm_diffs((u, v), self.res1)

        # print("Pos b", best_match_uv)
        if self._config["norm_by_descr_dim"]:
            norm_diffs_2 = norm_diffs_2 / np.sqrt(D) * 3
            norm_diffs_1 = norm_diffs_1 / np.sqrt(D) * 3

        threshold = self._config["norm_diff_threshold"]

        heatmap_color_1 = compute_gaussian_kernel_heatmap_from_norm_diffs(
            norm_diffs_1, self._config['kernel_variance'])

        heatmap_color_2 = compute_gaussian_kernel_heatmap_from_norm_diffs(
            norm_diffs_2, self._config['kernel_variance'])

        reticle_color = \
            COLOR_RED if best_match_diff < threshold else COLOR_BLACK

        draw_reticle(heatmap_color_1, u, v, reticle_color)

        draw_reticle(
            heatmap_color_2, best_match_uv[0], best_match_uv[1], reticle_color)

        draw_reticle(img_2_with_reticle,
                     best_match_uv[0], best_match_uv[1], reticle_color)
        blended_1 = cv2.addWeighted(self.img1, alpha, heatmap_color_1, beta, 0)
        blended_2 = cv2.addWeighted(self.img2, alpha, heatmap_color_2, beta, 0)

        embed_1 = scale(self.res1, out_range=(0, 255)).astype(np.uint8)
        embed_1 = embed_1[:, :, :3]  # for high dim descriptor, drop later dims
        embed_1 = np.ascontiguousarray(embed_1, dtype=np.uint8)
        draw_reticle(embed_1, u, v, COLOR_GREEN)

        embed_2 = scale(self.res2, out_range=(0, 255)).astype(np.uint8)
        embed_2 = embed_2[:, :, :3]  # for high dim descriptor, drop later dims
        embed_2 = np.ascontiguousarray(embed_2, dtype=np.uint8)
        draw_reticle(embed_2, best_match_uv[0],
                     best_match_uv[1], reticle_color)

        display((img_1_with_reticle, img_2_with_reticle),
                # (embed_1, embed_2),
                (blended_1, blended_2))

        if event == cv2.EVENT_LBUTTONDOWN:
            info_dict = {
                "uv_a": (u, v),
                "uv_b": best_match_uv,
                "norm dist": best_match_diff,
            }
            print(info_dict)
            np.save("lid_header_norm_diffs.npy", norm_diffs_2)
            np.save("lid_header_rgb.npy", self.img2)

            # self.axes.hist(norm_diffs.flatten(), bins=50)
            # self.fig.savefig("hist_bef.png")
            # self.fig.canvas.draw()
            # self.fig.savefig("hist_aft.png")

    def run(self):
        # plt.ion()
        # self.fig = plt.figure()
        # self.axes = self.fig.add_subplot(111)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.find_best_match)

        self._get_new_images()
        self._compute_descriptors()

        while True:
            k = cv2.waitKey(40) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                self._get_new_images()
                self._compute_descriptors()
            elif k == ord('r'):
                self._get_new_second_image()
                self._compute_descriptors()
            elif k == ord('s'):
                self.tensor1, self.tensor2 = self.tensor2, self.tensor1
                self.img1, self.img2 = self.img2, self.img1
                self._compute_descriptors()
            elif k == ord('p'):
                if self._paused:
                    self._paused = False
                else:
                    self._paused = True

        cv2.destroyAllWindows()
