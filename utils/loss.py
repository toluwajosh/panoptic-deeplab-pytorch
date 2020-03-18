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
        semantic_predict, center_predict, x_reg_pred, y_reg_pred = prediction
        # x_reg_pred = center_regress_predict[:, 0, :, :]*60
        # y_reg_pred = center_regress_predict[:, 1, :, :]*50

        # Debug:
        # print(torch.min(center))
        # print(torch.max(center))
        # print(torch.min(x_reg))
        # print(torch.max(x_reg))
        # print(torch.min(y_reg))
        # print(torch.max(y_reg))
        # print()

        # print(torch.min(center_predict).data)
        # print(torch.max(center_predict).data)
        # print(torch.min(x_reg_pred).data)
        # print(torch.max(x_reg_pred).data)
        # print(torch.min(y_reg_pred).data)
        # print(torch.max(y_reg_pred).data)
        # exit(0)

        # calculate losses
        semantic_loss = self.semantic_loss(semantic_predict, label)
        center_loss = mse_loss(center_predict, center.unsqueeze(1))
        # center_loss = l1_loss(center_predict, center.unsqueeze(1))
        x_reg_loss = mse_loss(x_reg_pred, x_reg.unsqueeze(1))
        y_reg_loss = mse_loss(y_reg_pred, y_reg.unsqueeze(1))
        return (
            semantic_loss * 10.0,
            center_loss * 0.025,
            x_reg_loss * 0.01,
            y_reg_loss * 0.01,
        )


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
