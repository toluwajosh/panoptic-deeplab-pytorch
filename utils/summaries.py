import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(
        self,
        writer,
        dataset,
        image,
        target,
        output,
        global_step,
        centers=None,
        reg_x=None,
        reg_y=None,
    ):
        # reg_x = reg[:, 0:1, :, :]
        # reg_y = reg[:, 1:2, :, :]
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image("Image", grid_image, global_step)
        grid_image = make_grid(
            decode_seg_map_sequence(
                torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                dataset=dataset,
            ),
            3,
            normalize=False,
            range=(0, 255),
        )
        writer.add_image("Predicted label", grid_image, global_step)
        grid_image = make_grid(
            decode_seg_map_sequence(
                torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                dataset=dataset,
            ),
            3,
            normalize=False,
            range=(0, 255),
        )
        writer.add_image("Groundtruth label", grid_image, global_step)
        if centers is not None:
            grid_image = make_grid(
                centers[:3].clone().cpu().data, 3, normalize=True,
            )
            writer.add_image("Centers image", grid_image, global_step)

            grid_image = make_grid(
                reg_x[:3].clone().cpu().data, 3, normalize=True,
            )
            writer.add_image("reg_x image", grid_image, global_step)

            grid_image = make_grid(
                reg_y[:3].clone().cpu().data, 3, normalize=True,
            )
            writer.add_image("reg_y image", grid_image, global_step)
