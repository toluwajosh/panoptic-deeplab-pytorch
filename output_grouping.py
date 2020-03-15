"""Output Grouping for Panoptic Segmentation
"""


def to_panoptic(semantic, centers, x_reg, y_reg, foreground_mask=None):
    """Group outputs of model to give panoptic segmentation for each pixel

    Arguments:
        semantic {tensor} -- output of semantic segmentation
        centers {tensor} -- output of centers prediction
        x_reg {tensor} -- output of center regresssion in the x axis
        y_reg {tensro} -- output of center regression in the y axis

    Keyword Arguments:
        foreground_mask {tensor} -- Mask for foreground pixels (default: {None})
    """

    #

    return NotImplementedError
