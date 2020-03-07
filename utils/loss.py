import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


class SegmentationLosses(object):
    def __init__(
        self,
        weight=None,
        size_average=True,
        batch_average=True,
        ignore_index=255,
        cuda=False,
    ):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode="ce"):
        """Choices: ['ce' or 'focal']"""
        if mode == "ce":
            return self.CrossEntropyLoss
        elif mode == "focal":
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            size_average=self.size_average,
        )
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            size_average=self.size_average,
        )
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class PanopticLosses(object):
    def __init__(
        self,
        weight=None,
        size_average=True,
        batch_average=True,
        ignore_index=255,
        cuda=False,
    ):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        # by default
        self.semantic_loss = self.CrossEntropyLoss

    def build_loss(self, mode="ce"):
        """Choices: ['ce' or 'focal']"""
        if mode == "ce":
            self.semantic_loss = self.CrossEntropyLoss
            return self
        elif mode == "focal":
            self.semantic_loss = self.FocalLoss
            return self
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            weight=self.weight, ignore_index=self.ignore_index,
        )
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            size_average=self.size_average,
        )
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def forward(self, prediction, label, center, x_reg, y_reg):
        b, w, h = center.shape
        x_semantic, x_center, x_center_regress = prediction

        # # normalize targets
        # center = center / 255.0
        # x_reg = x_reg / 255.0
        # y_reg = y_reg / 255.0

        # # mask pixels for stuff categories
        # mask = torch.zeros_like(label)
        # mask[label > 0] = 1
        # x_center = x_center * mask.view(b, 1, w, h)
        # x_center_regress = x_center_regress * mask.view(b, 1, w, h)

        # calculate losses
        semantic_loss = self.semantic_loss(x_semantic, label)
        center_loss = mse_loss(x_center, center.unsqueeze(1))
        center_regress = torch.cat([x_reg.unsqueeze(1), y_reg.unsqueeze(1)], 1)
        center_regress_loss = l1_loss(x_center_regress, center_regress)
        return semantic_loss, center_loss , center_regress_loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
