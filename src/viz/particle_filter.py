import matplotlib.pyplot as plt
import torch

from viz.operations import np_channel_front2back, rgb2gray


class ParticleFilterViz:
    def __init__(self):
        self.batch = [0, 0, 0, 0]
        self.keypoints = [0, 1, 8, 9]
        self.n_rows = len(self.batch)
        self.n_cols = 3

        self.fig_names = ['kp{}'.format(str(i))
                          for i in range(len(self.keypoints))]

        self.cbars = []

        self.traj_counter = 0
        self.obs_counter = 0
        self.file = 'src/_tmp/pf-'

        self.show = False

    def reset_episode(self):
        self.traj_counter += 1
        self.obs_counter = 0

    def run(self):
        self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols,
                                           figsize=(19.2, 14.4))

        for i in range(self.n_rows):
            sps = self.axes[i][0].get_subplotspec()
            self.axes[i][0].remove()
            self.axes[i][0] = self.fig.add_subplot(sps, projection="3d")

        for i in range(self.n_rows):  # fill some dummy value to setup color bar
            ax = self.axes[i][0]
            scp = ax.scatter(0.5, 0.5, 0.5, c=0.5, cmap='hot_r')
            # scp.set_clim(0, 1)
            cb = plt.colorbar(scp, ax=ax)

            self.cbars.append(cb)

        plt.ion()
        plt.tight_layout()

        if self.show:
            plt.show()

    def update(self, coordinates, weights, prediction, heatmaps, keypoints_2d,
               rgbs):

        prediction = torch.stack(torch.chunk(prediction, 3, dim=1), dim=2)

        for i in range(self.n_rows):
            ax = self.axes[i][0]
            self.cbars[i].remove()
            ax.clear()

            b = self.batch[i]
            k = self.keypoints[i]

            c = coordinates[b][k]
            w = weights[b][k]
            kp3d = prediction[b][k]

            scp = ax.scatter(c[..., 0], c[..., 1], c[..., 2], c=w, cmap='hot_r')

            self.cbars[i] = plt.colorbar(scp, ax=ax)

            ax.scatter(kp3d[..., 0], kp3d[..., 1], kp3d[..., 2], marker="X",
                       c='b')

            for j, (h, r, kp2d) in enumerate(zip(heatmaps, rgbs, keypoints_2d)):  # noqa 501
                ax = self.axes[i][j + 1]
                ax.clear()
                img = rgb2gray(np_channel_front2back(r[b].numpy()))
                ax.imshow(img, cmap='gray', alpha=0.3, interpolation='none')
                ax.imshow(h[b][k], alpha=0.7, cmap='hot_r', interpolation='none')
                ax.scatter(kp2d[b][k][0], kp2d[b][k][1], marker="X", c='b')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        file = self.file + str(self.traj_counter) + "-" + \
            str(self.obs_counter) + '.png'
        plt.savefig(file)

        self.obs_counter += 1

        if self.show:
            plt.pause(.001)
            self.fig.canvas.draw()
