"""
Encoder Factory for UniCeption
"""

import os

from uniception.models.encoders.base import (
    EncoderGlobalRepInput,
    EncoderInput,
    UniCeptionEncoderBase,
    UniCeptionViTEncoderBase,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
    ViTEncoderOutput,
)

# Define encoder configurations
ENCODER_CONFIGS = {
    "croco": {
        "class": ("uniception.models.encoders.croco", "CroCoEncoder"),
        "intermediate_feature_returner_class": ("uniception.models.encoders.croco", "CroCoIntermediateFeatureReturner"),
        "supported_models": ["CroCov2", "DUSt3R", "MASt3R"],
    },
    "dense_rep_encoder": {
        "class": ("uniception.models.encoders.dense_rep_encoder", "DenseRepresentationEncoder"),
        "supported_models": ["Dense-Representation-Encoder"],
    },
    "dinov2": {
        "class": ("uniception.models.encoders.dinov2", "DINOv2Encoder"),
        "intermediate_feature_returner_class": ("uniception.models.encoders.dinov2", "DINOv2IntermediateFeatureReturner"),
        "supported_models": ["DINOv2", "DINOv2-Registers", "DINOv2-Depth-Anythingv2"],
    },
    "dinov3": {
        "class": ("uniception.models.encoders.dinov3", "DINOv3Encoder"),
        "intermediate_feature_returner_class": ("uniception.models.encoders.dinov3", "DINOv3IntermediateFeatureReturner"),
        "supported_models": ["DINOv3-ViT"],
    },
    "dinov3_convnext": {
        "class": ("uniception.models.encoders.dinov3", "DINOv3ConvNextEncoder"),
        "intermediate_feature_returner_class": ("uniception.models.encoders.dinov3", "DINOv3ConvNextIntermediateFeatureReturner"),
        "supported_models": ["DINOv3-ConvNext"],
    },
    "dune": {
        "class": ("uniception.models.encoders.dune", "DUNEEncoder"),
        "intermediate_feature_returner_class": ("uniception.models.encoders.dune", "DUNEIntermediateFeatureReturner"),
        "supported_models": ["DUNE"],
    },
    "global_rep_encoder": {
        "class": ("uniception.models.encoders.global_rep_encoder", "GlobalRepresentationEncoder"),
        "supported_models": ["Global-Representation-Encoder"],
    },
    "patch_embedder": {
        "class": ("uniception.models.encoders.patch_embedder", "PatchEmbedder"),
        "supported_models": ["Patch-Embedder"],
    },
    "perception_encoder": {
        "class": ("uniception.models.encoders.perception_encoder", "PerceptionEncoder"),
        "intermediate_feature_returner_class": ("uniception.models.encoders.perception_encoder", "PerceptionEncoderIntermediateFeatureReturner"),
        "supported_models": ["Perception Encoder Core", "Preception Encoder Spatial"],
    },
    "radio": {
        "class": ("uniception.models.encoders.radio", "RADIOEncoder"),
        "intermediate_feature_returner_class": ("uniception.models.encoders.radio", "RADIOIntermediateFeatureReturner"),
        "supported_models": ["RADIO", "E-RADIO"],
    },
    "cosmos": {
        "class": ("uniception.models.encoders.cosmos", "CosmosEncoder"),
        "supported_models": ["Cosmos-Tokenizer CI8x8", "Cosmos-Tokenizer CI16x16"],
    },
    # Add other encoders here
}


def encoder_factory(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    """
    Encoder factory for UniCeption.
    Please use python3 -m uniception.models.encoders.list to see available encoders.

    Args:
        encoder_str (str): Name of the encoder to create.
        **kwargs: Additional keyword arguments to pass to the encoder constructor.

    Returns:
        UniCeptionEncoderBase: An instance of the specified encoder.
    """
    if encoder_str not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_str}. For valid encoder_str options, please use python3 -m uniception.models.encoders.list"
        )

    encoder_config = ENCODER_CONFIGS[encoder_str]
    from importlib import import_module
    module, name = encoder_config["class"]
    encoder_class = getattr(import_module(module), name)

    return encoder_class(**kwargs)


def feature_returner_encoder_factory(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    """
    Factory for UniCeption Encoders with support for intermediate feature returning.
    Please use python3 -m uniception.models.encoders.list to see available encoders.

    Args:
        encoder_str (str): Name of the encoder to create.
        **kwargs: Additional keyword arguments to pass to the encoder constructor.

    Returns:
        UniCeptionEncoderBase: An instance of the specified encoder.
    """
    if encoder_str not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_str}. For valid encoder_str options, please use python3 -m uniception.models.encoders.list"
        )

    encoder_config = ENCODER_CONFIGS[encoder_str]
    from importlib import import_module
    module, name = encoder_config["intermediate_feature_returner_class"]
    encoder_class = getattr(import_module(module), name)

    return encoder_class(**kwargs)


def get_available_encoders() -> list:
    """
    Get a list of available encoders in UniCeption.

    Returns:
        list: A list of available encoder names.
    """
    return list(ENCODER_CONFIGS.keys())


def print_available_encoder_models():
    """
    Print the currently supported encoders in UniCeption.
    """
    print("Currently Supported Encoders in UniCeption:\nFormat -> encoder_str: supported_models")
    for encoder_name, config in ENCODER_CONFIGS.items():
        print(f"{encoder_name}: {', '.join(config['supported_models'])}")


def _make_encoder_test(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    "Function to create encoders for testing purposes."
    current_file_path = os.path.abspath(__file__)
    relative_checkpoint_path = os.path.join(os.path.dirname(current_file_path), "../../../checkpoints/encoders")
    if encoder_str == "dummy":
        return UniCeptionEncoderBase(name="dummy", data_norm_type="dummy")
    elif encoder_str == "croco":
        return __getattr__("CroCoEncoder")(
            name="croco",
            data_norm_type="croco",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_224.pth",
            patch_embed_cls="PatchEmbedCroCo",
            **kwargs,
        )
    elif encoder_str == "dust3r_224":
        return __getattr__("CroCoEncoder")(
            name="dust3r_224",
            data_norm_type="dust3r",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_224_DUSt3R_linear.pth",
            patch_embed_cls="PatchEmbedDust3R",
            **kwargs,
        )
    elif encoder_str == "dust3r_512":
        return __getattr__("CroCoEncoder")(
            name="dust3r_512",
            data_norm_type="dust3r",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_512_DUSt3R_linear.pth",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
            **kwargs,
        )
    elif encoder_str == "dust3r_512_dpt":
        return __getattr__("CroCoEncoder")(
            name="dust3r_512_dpt",
            data_norm_type="dust3r",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_512_DUSt3R_dpt.pth",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
            **kwargs,
        )
    elif encoder_str == "mast3r_512":
        return __getattr__("CroCoEncoder")(
            name="mast3r_512",
            data_norm_type="dust3r",
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/CroCo_Encoder_512_MASt3R.pth",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
            **kwargs,
        )
    elif "dinov2" in encoder_str:
        size = encoder_str.split("_")[1]
        size_single_cap_letter = size[0].upper()
        if "reg" in encoder_str:
            with_registers = True
            pretrained_checkpoint_path = None
        elif "dav2" in encoder_str:
            with_registers = False
            pretrained_checkpoint_path = (
                f"{relative_checkpoint_path}/DINOv2_ViT{size_single_cap_letter}_DepthAnythingV2.pth"
            )
        else:
            with_registers = False
            pretrained_checkpoint_path = None
        return __getattr__("DINOv2Encoder")(
            name=encoder_str.replace("_reg", ""),
            size=size,
            with_registers=with_registers,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            **kwargs,
        )
    elif "radio" in encoder_str:
        if "e-radio" in encoder_str:
            eradio_input_shape = (224, 224)
        else:
            eradio_input_shape = None
        return __getattr__("RADIOEncoder")(
            name=encoder_str,
            model_version=encoder_str,
            eradio_input_shape=eradio_input_shape,
            **kwargs,
        )
    elif "dinov3_convnext" in encoder_str:
        size = encoder_str.split("_")[-1]
        dinov3_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../local/dinov3"))
        return __getattr__("DINOv3ConvNextEncoder")(
            name=encoder_str,
            dinov3_repo_dir=dinov3_repo_dir,
            size=size,
            weights=f"{relative_checkpoint_path}/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
            **kwargs,
        )
    elif "dinov3" in encoder_str:
        size = encoder_str.split("_")[-1]
        dinov3_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../local/dinov3"))
        return __getattr__("DINOv3Encoder")(
            name=encoder_str,
            dinov3_repo_dir=dinov3_repo_dir,
            size=size,
            weights=f"{relative_checkpoint_path}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            **kwargs,
        )
    elif "dune" in encoder_str:
        return __getattr__("DUNEEncoder")(
            name="dune_base",
            vit_size="base",
            pe_image_size=448,
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/dune_vitbase14_448.pth",
            **kwargs,
        )
    elif "cosmos" in encoder_str:
        patch_size = int(encoder_str.split("x")[-1])
        return __getattr__("CosmosEncoder")(
            name=encoder_str,
            patch_size=patch_size,
            pretrained_checkpoint_path=f"{relative_checkpoint_path}/Cosmos-Tokenizer-CI{patch_size}x{patch_size}/encoder.pth",
            **kwargs,
        )
    elif "perception_encoder" in encoder_str:
        pe_type = encoder_str.split("_")[2]
        pe_size = encoder_str.split("_")[3]
        return __getattr__("PerceptionEncoder")(
            name=encoder_str,
            patch_size=16 if pe_size == "base" else 14,
            data_norm_type="perception_encoder",
            size=pe_size,
            checkpoint_type=pe_type,
        )
    elif "patch_embedder" in encoder_str:
        return __getattr__("PatchEmbedder")(name=encoder_str, **kwargs)
    else:
        raise ValueError(f"Unknown encoder: {encoder_str}")


__all__ = [
    "encoder_factory",
    "get_available_encoders",
    "print_available_encoder_models",
    "_make_encoder_test",
    "UniCeptionEncoderBase",
    "UniCeptionViTEncoderBase",
    "EncoderInput",
    "ViTEncoderInput",
    "ViTEncoderOutput",
]

_ENCODER_CLASSES = {'CosmosEncoder': ('uniception.models.encoders.cosmos', 'CosmosEncoder'), 'CroCoEncoder': ('uniception.models.encoders.croco', 'CroCoEncoder'), 'CroCoIntermediateFeatureReturner': ('uniception.models.encoders.croco', 'CroCoIntermediateFeatureReturner'), 'DenseRepresentationEncoder': ('uniception.models.encoders.dense_rep_encoder', 'DenseRepresentationEncoder'), 'DINOv2Encoder': ('uniception.models.encoders.dinov2', 'DINOv2Encoder'), 'DINOv2IntermediateFeatureReturner': ('uniception.models.encoders.dinov2', 'DINOv2IntermediateFeatureReturner'), 'DINOv3ConvNextEncoder': ('uniception.models.encoders.dinov3', 'DINOv3ConvNextEncoder'), 'DINOv3ConvNextIntermediateFeatureReturner': ('uniception.models.encoders.dinov3', 'DINOv3ConvNextIntermediateFeatureReturner'), 'DINOv3Encoder': ('uniception.models.encoders.dinov3', 'DINOv3Encoder'), 'DINOv3IntermediateFeatureReturner': ('uniception.models.encoders.dinov3', 'DINOv3IntermediateFeatureReturner'), 'DUNEEncoder': ('uniception.models.encoders.dune', 'DUNEEncoder'), 'DUNEIntermediateFeatureReturner': ('uniception.models.encoders.dune', 'DUNEIntermediateFeatureReturner'), 'GlobalRepresentationEncoder': ('uniception.models.encoders.global_rep_encoder', 'GlobalRepresentationEncoder'), 'PatchEmbedder': ('uniception.models.encoders.patch_embedder', 'PatchEmbedder'), 'PerceptionEncoder': ('uniception.models.encoders.perception_encoder', 'PerceptionEncoder'), 'PerceptionEncoderIntermediateFeatureReturner': ('uniception.models.encoders.perception_encoder', 'PerceptionEncoderIntermediateFeatureReturner'), 'RADIOEncoder': ('uniception.models.encoders.radio', 'RADIOEncoder'), 'RADIOIntermediateFeatureReturner': ('uniception.models.encoders.radio', 'RADIOIntermediateFeatureReturner')}

def __getattr__(name):
    if name not in _ENCODER_CLASSES:
        raise AttributeError(name)
    from importlib import import_module
    module, cls = _ENCODER_CLASSES[name]
    value = getattr(import_module(module), cls)
    globals()[name] = value
    return value
