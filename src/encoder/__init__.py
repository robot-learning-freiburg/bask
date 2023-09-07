from encoder.bvae import BVAE
from encoder.cnn import CNN, CNNDepth
from encoder.keypoints import KeypointsPredictor
from encoder.keypoints_gt import GTKeypointsPredictor
from encoder.monet import Monet


encoder_switch = {
    "bvae": BVAE,
    "monet": Monet,
    "keypoints": KeypointsPredictor,
    "keypoints_gt": GTKeypointsPredictor,
    "cnn": CNN,
    "cnnd": CNNDepth,
}

encoder_names = list(encoder_switch.keys())
