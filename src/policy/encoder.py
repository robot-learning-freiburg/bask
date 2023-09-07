import torch
from loguru import logger

from dataset.bc import BCDataset
from encoder import encoder_switch
from env.environment import BaseEnvironment
from models.keypoints.keypoints import KeypointsTypes as KeypointsTypes
from policy.policy import Policy
from utils.logging import indent_func_log, indent_logs, log_constructor
from utils.misc import get_and_log_failure as get_conf
from utils.select_gpu import device


class EncoderPolicy(Policy):
    """
    Policy that uses an encoder to encode the camera observation.
    """
    @log_constructor
    def __init__(self, config: dict, encoder_checkpoint: str | None = None,
                 **kwargs) -> None:

        self.config = config

        Encoder = encoder_switch[config["encoder"]]

        embedding_dim = Encoder.get_latent_dim(
            config['encoder_config'].get("encoder"), n_cams=config['n_cams'],
            image_dim=config['encoder_config']['obs_config']['image_dim'])
        config["visual_embedding_dim"] = embedding_dim
        logger.info("Embedding dim: {}", embedding_dim)

        super().__init__(config)

        self.encoder = Encoder(config["encoder_config"]).to(device)

        # The encoder is a sub-module of the encoder policy. So in eval we do
        # not need to read the encoder from disk before reading the policy.
        if encoder_checkpoint is not None:
            logger.info("Loading encoder checkpoint from {}",
                        encoder_checkpoint)
            self.encoder.from_disk(encoder_checkpoint)
            if config.get("end-to-end") or \
                    config["encoder_config"].get("end-to-end"):
                logger.info("Adding encoder to optim params.")
                self.optimizer.add_param_group(
                    {'params': self.encoder.parameters()})
            else:
                self.encoder.requires_grad_(False)
                self.encoder.eval()

    def initialize_parameters_via_dataset(self, replay_memory: BCDataset,
                                          cameras: tuple[str]) -> None:
        # eg. reference descriptor selection for keypoints model
        self.encoder.initialize_parameters_via_dataset(
            replay_memory, cam=cameras[0])

        if self.config["encoder"] == "keypoints":
            if self.config["encoder_config"]["encoder"]["keypoints"]["type"] \
                is KeypointsTypes.ODS and not (
                    self.config.get("end-to-end") or
                    self.config["encoder_config"].get("end-to-end")):
                logger.info("Adding reference descriptor to optim params.")
                self.optimizer.add_param_group(
                    {'params': self.encoder._reference_descriptor_vec})

    def from_disk(self, chekpoint_path: str) -> None:
        state_dict = torch.load(chekpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys: {}".format(missing))
        if unexpected:
            logger.warning("Unexpected keys: {}".format(unexpected))
        self.encoder.requires_grad_(False)  # TODO: enable (fine-)tuning?
        self.encoder.eval()

        if self.config["encoder"] == "keypoints":
            self.encoder.set_model_image_normalization()

    def reset_episode(self, env: BaseEnvironment | None = None) -> None:
        self.encoder.reset_episode()


class EncoderPseudoPolicy():
    """
    EncoderPolicy variant for loading pre-computed embeddings from disk.
    This is only a PseudoPolicy, though, as it does not implement full policy
    functionality. It is only used to wrap the encoder during kp pre-encoding.
    Manages encoder checkpoint loading and saving, as well as data-init of
    encoder.
    """
    @log_constructor
    def __init__(self, config: dict, encoder_checkpoint: str | None = None,
                 copy_selection_from: str | None = None, **kwargs) -> None:

        self.config = config

        Encoder = encoder_switch[config["encoder"]]
        embedding_dim = Encoder.get_latent_dim(
            config['encoder_config']["encoder"], n_cams=config['n_cams'])
        config["visual_embedding_dim"] = embedding_dim
        logger.info("Embedding dim: {}", embedding_dim)

        self.encoder = Encoder(config["encoder_config"])

        self.encoder_checkpoint = encoder_checkpoint
        self.copy_selection_from = copy_selection_from

        # When KPs are already selected, we do not need to initialize via the
        # dataset anymore.
        self.skip_dataset_init = get_conf(config, "kp_pre_encoded", False)

    def initialize_parameters_via_dataset(self, replay_memory: BCDataset,
                                          cameras: tuple[str]) -> None:
        ckpt = self.copy_selection_from

        if self.skip_dataset_init:
            logger.info("Skipping dataset initialization.")
        elif ckpt is None:
            self.encoder.initialize_parameters_via_dataset(
                replay_memory, cam=cameras[0])
        else:
            self.copy_reference_from_disk(ckpt, replay_memory, cameras)

    def copy_reference_from_disk(self, ckpt: str, replay_memory: BCDataset,
                                 cameras: tuple[str]) -> None:
        logger.info("Copying reference positions and descriptors from {}",
                    ckpt)
        state_dict = torch.load(ckpt, map_location='cpu')

        try:
            self.encoder.ref_pixels_uv = state_dict["ref_pixels_uv"].to(
                device)
            is_policy_checkpoint = False
        except KeyError:
            self.encoder.ref_pixels_uv = \
                state_dict['encoder.ref_pixels_uv'].to(device)
            is_policy_checkpoint = True

        try:  # for other kp encoder
            self.encoder._reference_descriptor_vec = state_dict[
                "encoder._reference_descriptor_vec" if is_policy_checkpoint \
                else "_reference_descriptor_vec"].to(device)

        except KeyError:  # in gt_kp encoder ref descriptor does not exist
            with indent_logs():
                logger.info("Got GT model. Reconstructing ref descriptor.")
            self.encoder._reference_descriptor_vec = \
                self.encoder.reconstruct_ref_descriptor_from_gt(
                    replay_memory, state_dict["ref_pixels_uv"],
                    state_dict["ref_object_poses"],
                    cam=cameras[0])
            # set for better prior for particle filter
            try:
                self.encoder.filter.ref_pixel_world = \
                    state_dict["ref_pixel_world"]
            except AttributeError:
                pass
            except Exception as e:
                raise e

    def encoder_to_disk(self, checkpoint_path: str) -> None:
        logger.info("Saving encoder:")

        assert (ckpt := self.encoder_checkpoint) is not None
        with indent_logs():
            logger.info("Adding encoder checkpoint to snapshot.", ckpt)
            self.encoder.from_disk(
                ckpt, ignore=['_reference_descriptor_vec', 'ref_pixels_uv'])

            self.encoder.requires_grad_(False)
            self.encoder.eval()

            self.encoder.to_disk(checkpoint_path)

    def reset_episode(self, env: BaseEnvironment | None = None) -> None:
        self.encoder.reset_episode()


class DiskReadEncoderPolicy(EncoderPseudoPolicy, EncoderPolicy):
    """
    EncoderPolicy for using pre-computed embeddings from disk during policy
    learning.
    On top of regular policy functionality, also manages encoder checkpoint
    loading and saving. Data-init of encoder is not needed, as this is taken
    care of during pre-encoding.
    """
    @log_constructor
    def __init__(self, config: dict, encoder_checkpoint: str | None = None,
                 **kwargs) -> None:

        # Init Module for assignment of encoder in EncoderPseudoPolicy init
        torch.nn.Module.__init__(self)

        EncoderPseudoPolicy.__init__(self, config, encoder_checkpoint, None)

        # Skip Module init here, as the encoder is not defined in Policy and
        # thus would get lost upon Module init.
        Policy.__init__(self, config, skip_module_init=True)


    def to_disk(self, file_name: str) -> None:
        logger.info("Saving policy:")

        assert (ckpt := self.encoder_checkpoint) is not None

        with indent_logs():
            logger.info("Adding encoder checkpoint to snapshot.")
            if self.config["encoder"] == "keypoints_gt":
                self.encoder.from_disk(ckpt, force_read=True)
            else:
                self.encoder.from_disk(ckpt)

            self.encoder.requires_grad_(False)
            self.encoder.eval()

            Policy.to_disk(self, file_name)
