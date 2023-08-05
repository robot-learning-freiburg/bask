import numpy as np
from pbdlib.mvn import MVN
from riepybdlib.statistics import Gaussian as MVNRBD
from scipy.linalg import block_diag

from utils.misc import multiply_iterable


def concat_mvn(gaussians):
    mvn = MVN()
    mvn.mu = np.concatenate([g.mu for g in gaussians])
    mvn._sigma = block_diag(*[g.sigma for g in gaussians])
    mvn._lmbda = block_diag(*[g.lmbda for g in gaussians])

    return mvn

def concat_mvn_rbd(gaussians):
    raise NotImplementedError
    manis = [g.manifold for g in gaussians]
    joint_manifold = multiply_iterable(manis)
    print(type(gaussians[0].mu), type(gaussians[0].sigma))
    joint_mu = np.concatenate([g.mu for g in gaussians])
    mvn = MVNRBD(joint_manifold, joint_mu, joint_sigma)
