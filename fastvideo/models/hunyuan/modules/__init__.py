from .models import HUNYUAN_VIDEO_CONFIG, HYVideoDiffusionTransformer
from .model_audio import HYVideoDiffusionTransformerAudio
from .model_audio_i2v import HYVideoDiffusionTransformerAudioI2V

def load_model(args, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    if args.model_type == "hunyuan_audio":
        model = HYVideoDiffusionTransformerAudio(
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
        return model
    elif args.model_type == "hunyuan_audio_i2v":
        model = HYVideoDiffusionTransformerAudioI2V(
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
        return model
    elif args.model in HUNYUAN_VIDEO_CONFIG.keys():
        model = HYVideoDiffusionTransformer(
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
        return model

    else:
        raise NotImplementedError()
