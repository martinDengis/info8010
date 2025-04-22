from .bibnet import BibNet
from .loss import BboxLoss

def build_bibnet(cfg):
    """Builds a BibNet model based on the provided configuration.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.

    Returns:
        BibNet: An instance of the BibNet model.
    """
    model = BibNet(cfg)
    return model

__all__ = ['BboxLoss', 'build_bibnet']