import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.aspp import build_aspp
from modeling.backbone import build_backbone
from modeling.panoptic_decoder import build_decoder as build_decoder
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class PanopticDeepLab(nn.Module):
    def __init__(
        self,
        backbone="resnet",
        output_stride=16,
        num_classes=21,
        sync_bn=True,
        freeze_bn=False,
    ):
        super(PanopticDeepLab, self).__init__()
        if backbone == "drn":
            output_stride = 8

        if sync_bn is True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        # TODO(toluwajosh): we should probably be able to use
        # the same backbone context for both semantic and instance heads
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        # semantic decoder
        self.semantic_decoder = build_decoder(256, backbone, BatchNorm)
        # TODO(toluwajosh): instance decoder
        self.panoptic_decoder = build_decoder(128, backbone, BatchNorm)

        self.semantic_predict = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=5, stride=1, padding=1, bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                256,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.instance_center_predict = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=1, bias=False),
            BatchNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=1, bias=False),
        )

        self.instance_center_regress = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=1, bias=False),
            BatchNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=1, bias=False),
        )

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, mid_level_feat, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        # semantic head
        x_semantic = self.semantic_decoder(x, mid_level_feat, low_level_feat)
        # panoptic head
        x_panoptic = self.panoptic_decoder(x, mid_level_feat, low_level_feat)

        # TODO(toluwajosh): make sure next stage is necessary
        x_semantic = F.interpolate(
            x_semantic,
            size=input.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        x_panoptic = F.interpolate(
            x_panoptic,
            size=input.size()[2:],
            mode="bilinear",
            align_corners=True,
        )

        x_semantic = self.semantic_predict(x_semantic)
        x_center_predict = self.instance_center_predict(x_panoptic)
        x_center_regress = self.instance_center_regress(x_panoptic)
        return x_semantic, x_center_predict, x_center_regress

    # TODO(toluwajosh): resolve the conflict
    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, SynchronizedBatchNorm2d):
    #             m.eval()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.eval()

    # # TODO(toluwajosh):
    # def get_1x_lr_params(self):
    #     modules = [self.backbone]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if self.freeze_bn:
    #                 if isinstance(m[1], nn.Conv2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p
    #             else:
    #                 if (
    #                     isinstance(m[1], nn.Conv2d)
    #                     or isinstance(m[1], SynchronizedBatchNorm2d)
    #                     or isinstance(m[1], nn.BatchNorm2d)
    #                 ):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p

    # def get_10x_lr_params(self):
    #     modules = [self.aspp, self.decoder]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if self.freeze_bn:
    #                 if isinstance(m[1], nn.Conv2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p
    #             else:
    #                 if (
    #                     isinstance(m[1], nn.Conv2d)
    #                     or isinstance(m[1], SynchronizedBatchNorm2d)
    #                     or isinstance(m[1], nn.BatchNorm2d)
    #                 ):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p


if __name__ == "__main__":
    model = DeepLab(backbone="mobilenet", output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
