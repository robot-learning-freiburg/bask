from encoder.bvae import BVAE
from encoder.cnn import CNN, CNNDepth
from encoder.keypoints import KeypointsPredictor
from encoder.keypoints_gt import GTKeypointsPredictor
from encoder.keypoints_var import VarKeypointsPredictor
from encoder.monet import Monet
from encoder.transporter import Transporter
from encoder.vit_extractor import VitFeatureEncoder

encoder_switch = {
    "transporter": Transporter,
    "bvae": BVAE,
    "monet": Monet,
    "keypoints": KeypointsPredictor,
    "keypoints_gt": GTKeypointsPredictor,
    "keypoints_var": VarKeypointsPredictor,
    "cnn": CNN,
    "cnnd": CNNDepth,
    "vit_extractor": VitFeatureEncoder,
}
