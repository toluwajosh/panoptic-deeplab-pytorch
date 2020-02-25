import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == "mobilenet_3stage":
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        in_ch_1 = 256
        out_ch_1 = 256
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                256,  # output from the aspp block
                out_ch_1,
                kernel_size=1,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm(out_ch_1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        in_ch_2 = 256 + low_level_inplanes
        out_ch_2 = 256
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_ch_2,
                out_ch_2,
                kernel_size=5,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm(out_ch_2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        in_ch_3 = 256 + 32
        out_ch_3 = num_classes
        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_ch_3,
                out_ch_3,
                kernel_size=5,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm(out_ch_3),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self._init_weight()

    def forward(self, x, mid_level_feat, low_level_feat):
        """
        expectation
        low level feat shape:  torch.Size([4, 24, 129, 129])
        x shape after aspp:  torch.Size([4, 256, 33, 33])
        x shape after decoder:  torch.Size([4, 21, 129, 129])
        x shape final:  torch.Size([4, 21, 513, 513])
        """
        x = self.conv_1(x)
        x = F.interpolate(
            x,
            size=low_level_feat.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.conv_2(x)
        x = F.interpolate(
            x,
            size=mid_level_feat.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat((x, mid_level_feat), dim=1)
        x = self.conv_3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
