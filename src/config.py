from dataset.dc import DcDatasetDataType
from encoder.keypoints import LRScheduleTypes, PriorTypes, ProjectionTypes
from models.keypoints.keypoints import KeypointsTypes

camera_pose = {
    "overhead": [0.2, 0, 0.2, 7.7486e-07, -0.194001, 7.7486e-07, 0.981001]
}

policy_config = {}

encoder_configs = {
    "dummy": None,
    "cnn": {
        "end-to-end": True
        },
    "cnnd": {
        "end-to-end": True
        },
    "transporter": {
        "pretrain": {
            "dataset_config": {},
            "training_config": {
                "lr": 5e-4,
                # TODO: Adam:  gradient-norm clipped to 1.0,
                # learning rate of 1e-3 decayed by 0.95 every 100k steps
            },
        },

        "training": {},

        "encoder": {
            "image_channels": 3,
            "k": 4,
            "n_keypoints": 10,
            "keypoint_std": 0.1,
            "architecture": {
                "image_encoder": {
                    "no_filters": (32, 32, 64, 128),
                    "stride": (1, 1, 2, 1),
                    "filter_size": (7, 3, 3, 3),
                    "padding": (3, 1, 1, 1)
                },
                "keypoint_encoder": {
                    "no_filters": (32, 32, 64, 128),
                    "stride": (1, 1, 2, 1),
                    "filter_size": (7, 3, 3, 3),
                    "padding": (3, 1, 1, 1)
                },
                "refinement_net": {
                    # NOTE constructed as inverse of encoder!
                },
            }
        },
    },
    "bvae": {
        "pretrain": {
            "dataset_config": {},
            "training_config": {
                "lr": 1e-4,

                "beta": 1.5,
                "kld_correction": True,  # weight KLD by additional M/N factor.
                # =====================================================================
                "loss_type": 'H',  # seems to be the one from the paper, other choice B
                "gamma": 10.0,  # these args only matter for the B loss_type -> ignore
                "max_capacity": 25,
                "Capacity_max_iter": 1e5,
                # =====================================================================
            },
        },

        "training": {},

        "encoder": {
            "image_channels": 3,
            "latent_dim": 32,  # 2, 16 also tried
            # archi args are for the encoder. decoder is reversed + final layer.
            # last one is FC, before: conv2d
            "hidden_dims": [32, 32, 32, 32, 128],
            "stride": (2, 2, 2, 2),
            "filter_size": (4, 4, 4, 4),
            "padding": (1, 1, 1, 1),
        },
    },
    "monet": {
        "pretrain": {
            "dataset_config": {},
            "training_config": {
                "lr": 1e-4,
                "beta": 5,  # weight for the encoder KLD
                "gamma": 0.5,  # not mentioned in paper?? taken from implementation, is weight of mask KLD
            },
        },

        "training": {},

        "encoder": {
            "image_channels": 3,
            # defined per bvae -> also gives no of bvaes
            "latent_dims": [16, 16],
            # maps object slots to bvae. Keys need to be range 0 to no_slots
            "slots": {**{0: 0}, **{i+1: 1 for i in range(5)}},
            "sigma_fg": 0.12,
            "sigma_bg": 0.09,  # not mentioned in paper, but original monet paper
        },
    },
    "keypoints": {
        "pretrain": {
            "dataset_config": {
                "batch_size": 16,
                "eval_batchsize": 160,

                "debug": False,
                "domain_randomize": True,
                "random_crop": True,
                "random_flip": True,
                "sample_matches_only_off_mask": True,
                "num_matching_attempts": 10000,
                "num_non_matches_per_match": 150,
                "_use_image_b_mask_inv": True,
                "fraction_masked_non_matches": 0.5,
                "cross_scene_num_samples": 10000,  # for different/same obj cross scene
                "data_type_probabilities": {
                    DcDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE: 1,
                    DcDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE: 0,
                    DcDatasetDataType.DIFFERENT_OBJECT: 0,
                    DcDatasetDataType.MULTI_OBJECT: 0,
                    DcDatasetDataType.SYNTHETIC_MULTI_OBJECT: 0
                },
            },
            "training_config": {
                "steps": 100,
                "eval_freq": 2,

                "lr": 1e-4,
                "lr_schedule": LRScheduleTypes.STEP,
                "weight_decay": 1e-4,
                "learning_rate_decay": 0.9,
                "steps_between_learning_rate_decay": 25,

                "no_samples_normalization": 100,
                "manual_kp_selection": True,

                "loss_function": {
                    "M_masked": 0.5,  # margin for masked non-match descriptor distance
                    "M_background": 0.5,  # margin for background
                    "M_pixel": 25,  # Clamp for pixel distance
                    "match_loss_weight": 1.0,
                    "non_match_loss_weight": 1.0,
                    "use_l2_pixel_loss_on_masked_non_matches": False,
                    "use_l2_pixel_loss_on_background_non_matches": False,
                    "scale_by_hard_negatives": True,
                    "scale_by_hard_negatives_DIFFERENT_OBJECT": True,
                    "alpha_triplet": 0.1,
                    "norm_by_descriptor_dim": True
                },
            },
        },

        "training": {
            "debug": True,  # TODO: this is an encoder key?
        },

        "encoder": {
            "normalize_images": True,
            "use_spatial_expectation": False,  # else uses the mode of the pdf
            "threshold_keypoint_dist": None,
            "overshadow_keypoints": False,  # for stereo cam setup only

            "descriptor_dim": 3,  # 2
            "projection": ProjectionTypes.GLOBAL_HARD,
            "vision_net": 'Resnet50_8s',  # Resnet101_8s
            # "dynamics_hidden_dims": [500, 500],
            "prior_type": PriorTypes.NONE,
            "taper_sm": 1,
            "use_motion_model": False,
            "motion_model_noisy": False,
            "motion_model_kernel": 9,
            "motion_model_sigma": 2,
            "keypoints": {
                "type": KeypointsTypes.SD,
                "n_sample": 16,  # larger when using WSD, eg. 100
                "n_keep": 5,  # only for SDS, WSDS
                # TODO other keys for WDS
            },
        },
    },
    "keypoints_gt": {
        "pretrain": {
            "training_config": {
                "manual_kp_selection": True,
            }
        },
        "encoder": {
            "debug": False,
            "descriptor_dim": 3,  # DUMMY value
            "projection": ProjectionTypes.EGO,
            "keypoints": {
                "type": KeypointsTypes.SD,
                "n_sample": 16,  # larger when using WSD, eg. 100
                "n_keep": 5,  # only for SDS, WSDS
            },
        },
    },
    "vit_extractor": {
        "encoder": {
            "load_size": None,
            "layer": 12,
            "facet": 'token ',  # 'key', 'query', 'value', 'token'
            "bin": False,
            "thresh": 0.05,
            "vision_net": 'dino_vits8',  # 'vit_base_patch8_224',
            "stride": 8,
        },
        "pretrain": {  # values for dc heatmap
            "dataset_config": {
                "data_type_probabilities": {
                    DcDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE: 1,
                    DcDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE: 0,
                    DcDatasetDataType.DIFFERENT_OBJECT: 0,
                    DcDatasetDataType.MULTI_OBJECT: 0,
                    DcDatasetDataType.SYNTHETIC_MULTI_OBJECT: 0
                },
            },
            "training_config": {
                "loss_function": {
                    "norm_by_descriptor_dim": True
                },
            },
        },
    }
}
