import torch
from loguru import logger

from encoder import encoder_switch
from encoder.disk_read import DiskReadEncoder
from encoder.keypoints import KeypointsPredictor
from models.keypoints.keypoints import KeypointsTypes as KeypointsTypes
from policy.policy import Policy
from utils.select_gpu import device


class EncoderPolicy(Policy):
    def __init__(self, config, encoder_checkpoint=None, **kwargs):
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
                # TODO: have own optimizer?
            else:
                self.encoder.requires_grad_(False)
                self.encoder.eval()

    def initialize_parameters_via_dataset(self, replay_memory):
        # eg. reference descriptor selection for keypoints model
        init_encoder = getattr(self.encoder,
                               "initialize_parameters_via_dataset", None)
        if callable(init_encoder):
            init_encoder(replay_memory)
        else:
            logger.info("This policy does not use dataset initialization.")

        if self.config["encoder"] == "keypoints":
            if self.config["encoder_config"]["encoder"]["keypoints"]["type"] \
                is KeypointsTypes.ODS and not (
                    self.config.get("end-to-end") or
                    self.config["encoder_config"].get("end-to-end")):
                logger.info("Adding reference descriptor to optim params.")
                self.optimizer.add_param_group(
                    {'params': self.encoder._reference_descriptor_vec})

    def from_disk(self, chekpoint_path):
        state_dict = torch.load(chekpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys: {}".format(missing))
        if unexpected:
            logger.warning("Unexpected keys: {}".format(unexpected))
        self.encoder.requires_grad_(False)  # TODO: enable (fine-)tuning?
        self.encoder.eval()

        if self.config["encoder"] in ["keypoints", "keypoints_var"]:
            self.encoder.set_model_image_normalization()


class PseudoPreEmbeddedEncoderPolicy():  # Policy):
    # Variant that does not encode online, but precomputes. Cheaper!
    # This is only a pseudo-policy to wrap the kp-encoder. For a real policy
    # see below.
    # For KPs: still need to select reference descriptors and compute their
    # location at each step.
    def __init__(self, config, encoder_checkpoint=None,
                 copy_selection_from=None, **kwargs):
        self.config = config

        self.PseudoEncoder = encoder_switch[config["encoder"]]
        embedding_dim = self.PseudoEncoder.get_latent_dim(
            config['encoder_config']["encoder"], n_cams=config['n_cams'])
        config["visual_embedding_dim"] = embedding_dim
        logger.info("Embedding dim: {}", embedding_dim)

        self.encoder = DiskReadEncoder(
            config["encoder_config"], self.PseudoEncoder)

        self.encoder_checkpoint = encoder_checkpoint
        self.copy_selection_from = copy_selection_from

    def initialize_parameters_via_dataset(self, replay_memory):
        if (ckpt := self.copy_selection_from) is None:
            init_encoder = getattr(self.encoder,
                                   "initialize_parameters_via_dataset", None)
            if callable(init_encoder):
                init_encoder(replay_memory)
            else:
                logger.info("This policy does not use dataset initialization.")
        else:
            logger.info("Copying reference positions and descriptors from {}",
                        ckpt)
            state_dict = torch.load(ckpt, map_location='cpu')
            try:
                self.encoder.ref_pixels_uv = state_dict["ref_pixels_uv"].to(device)
                is_policy_checkpoint = False
            except KeyError:
                self.encoder.ref_pixels_uv = state_dict['encoder.ref_pixels_uv'].to(device)
                is_policy_checkpoint = True

            try:  # for other kp encoder
                self.encoder._reference_descriptor_vec = state_dict[
                    "encoder._reference_descriptor_vec" if is_policy_checkpoint else "_reference_descriptor_vec"].to(device)
            except KeyError:  # in gt_kp encoder ref descriptor does not exist
                logger.info("  Got GT model. Reconstructing ref descriptor.")
                self.encoder._reference_descriptor_vec = \
                    self.encoder.reconstruct_ref_descriptor_from_gt(
                        replay_memory, state_dict["ref_pixels_uv"],
                        state_dict["ref_object_pose"])
                # set for better prior for particle filter
                try:
                    self.encoder.particle_filter.ref_pixel_world = \
                        state_dict["ref_pixel_world"]
                except AttributeError:
                    pass
                except Exception as e:
                    raise e

            # HACK. TODO: put this back into the disk_read.py or get rid off.
            self.encoder.EncoderClass.config = self.encoder.encoder_config
            self.encoder.EncoderClass.image_height, self.encoder.EncoderClass.image_width = \
                self.config['encoder_config']['obs_config']['image_dim']
                # self.encoder.encoder_config.get("image_size", (256, 256))
            self.encoder.EncoderClass.get_dc_dim()
            self.encoder.EncoderClass.setup_pixel_maps()

    def encodder_to_disk(self, checkpoint_path):
        dr_encoder = self.encoder
        # add the encoder to self for later use of the checkpoint
        encoder = self.PseudoEncoder(self.config["encoder_config"])
        assert (ckpt := self.encoder_checkpoint) is not None
        logger.info("Adding encoder checkpoint from {} to snapshot.", ckpt)
        encoder.from_disk(
            ckpt, ignore=['_reference_descriptor_vec', 'ref_pixels_uv'])

        encoder.requires_grad_(False)
        encoder.eval()

        encoder._reference_descriptor_vec = \
            dr_encoder._reference_descriptor_vec
        encoder.ref_pixels_uv = dr_encoder.ref_pixels_uv

        return encoder.to_disk(checkpoint_path)


class PreEmbeddedEncoderPolicy(PseudoPreEmbeddedEncoderPolicy, EncoderPolicy):
    def __init__(self, config, encoder_checkpoint=None, **kwargs):
        self.config = config

        self.PseudoEncoder = encoder_switch[config["encoder"]]
        embedding_dim = self.PseudoEncoder.get_latent_dim(
            config['encoder_config']["encoder"], n_cams=config['n_cams'])
        config["visual_embedding_dim"] = embedding_dim
        logger.info("Embedding dim: {}", embedding_dim)

        Policy.__init__(self, config)

        self.encoder = DiskReadEncoder(
            config["encoder_config"], self.PseudoEncoder)

        self.encoder_checkpoint = encoder_checkpoint
        self.copy_selection_from = None

    def to_disk(self, file_name):
        dr_encoder = self.encoder

        self.encoder = self.PseudoEncoder(self.config["encoder_config"])
        assert (ckpt := self.encoder_checkpoint) is not None
        logger.info("Adding encoder checkpoint from {} to snapshot.", ckpt)
        if self.config["encoder"] == "keypoints_gt":
            self.encoder.from_disk(ckpt, force_read=True)
        else:
            self.encoder.from_disk(ckpt)

        self.encoder.requires_grad_(False)
        self.encoder.eval()

        Policy.to_disk(self, file_name)

        self.encoder = dr_encoder  # restore to continue training


class KPCompleteProcomputeEncoderPolicy(Policy):
    # Variant of the PreEmbeddedEncoderPolicy that uses already compute kp
    # locations. For other encoders just use PreEmbeddedEncoderPolicy.
    def __init__(self, config, encoder_checkpoint=None, **kwargs):
        self.config = config

        Encoder = encoder_switch[config["encoder"]]
        embedding_dim = Encoder.get_latent_dim(
            config['encoder_config']["encoder"], n_cams=config['n_cams'])
        config["visual_embedding_dim"] = embedding_dim
        logger.info("Embedding dim: {}", embedding_dim)

        super().__init__(config)

        self.encoder = DiskReadEncoder(
            config["encoder_config"], KeypointsPredictor, attr_name='kp'
            ).to(device)

        self.encoder_checkpoint = encoder_checkpoint

    def initialize_parameters_via_dataset(self, replay_memory):
        logger.info("This policy does not use dataset initialization.")
        # TODO: need it for encoder snapshot?

    def to_disk(self, checkpoint_path):
        dr_encoder = self.encoder
        # add the encoder to self for later use of the checkpoint
        self.encoder = KeypointsPredictor(self.config["encoder_config"])
        assert (ckpt := self.encoder_checkpoint) is not None
        logger.info("Adding encoder checkpoint from {} to snapshot.", ckpt)
        self.encoder.from_disk(ckpt)

        self.encoder.requires_grad_(False)
        self.encoder.eval()

        Policy.to_disk(self, checkpoint_path)

        self.encoder = dr_encoder  # restore to continue training
