import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders import make_data_loader
from dataloaders.utils import decode_seg_map_sequence
from modeling.panoptic_deeplab import PanopticDeepLab
from modeling.sync_batchnorm.replicate import patch_replication_callback
from mypath import Path
from utils.calculate_weights import calculate_weigths_labels
from utils.loss import PanopticLosses, SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.summaries import TensorboardSummary


class Tester(object):
    NUM_CLASSES = 21

    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.nclass,
        ) = make_data_loader(args, **kwargs)

        # Define network
        model = PanopticDeepLab(
            num_classes=self.nclass,
            backbone=args.backbone,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
        )

        self.model = model

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_loader)
        )

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.args.gpu_ids
            )
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        if not os.path.isfile(args.resume):
            raise RuntimeError(
                "=> no checkpoint/model found at '{}'".format(args.resume)
            )
        checkpoint = torch.load(args.resume)
        if args.cuda:
            self.model.module.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint["state_dict"])
        self.best_pred = checkpoint["best_pred"]
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )

        self.void_classes = [
            # 0,
            1,
            2,
            3,
            4,
            5,
            6,
            9,
            10,
            14,
            15,
            16,
            18,
            29,
            # 30,
            -1,
        ]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
            30,  # added
            0,  # added
        ]
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "trailer",  # added
            "unlabelled",  # added
        ]

        # in label IDs
        self.things_category = [24, 25, 26, 27, 28, 31, 32, 33]

    def test_saving(self, epoch):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc="\r")
        test_loss = 0.0
        for i, (sample, filepath) in enumerate(tbar):
            image = sample["image"]

            if self.args.cuda:
                image = image.cuda()

            # print(filepath)
            new_filepath = filepath[0].replace(
                "gtFine_trainvaltest", "gtFine_test_result"
            )
            filename = new_filepath.split("/")[-1]
            print()
            directorypath = new_filepath.split("/")[:-1]
            directorypath = "/".join(directorypath)
            if not os.path.exists(directorypath):
                os.makedirs(directorypath)
                # print("directorypath: ", directorypath)

            input_image = image.clone()
            with torch.no_grad():
                try:
                    output = self.model(input_image)
                except ValueError as identifier:
                    # there was an error with wrong input size
                    print("Error: ", identifier)
                    continue

            semantic_pred, center_pred, x_offset_pred, y_offset_pred = output

            # ############## to create the InstanceIDs
            # 1. convert trainIDs to labelIDs
            semantic_labels = np.argmax(semantic_pred.cpu(), axis=1)
            for trainId in range(self.NUM_CLASSES):
                semantic_labels[
                    semantic_labels == trainId
                ] = self.valid_classes[trainId]
            # 2. multiply by 1000
            # print("semantic_labels.shape: ", semantic_labels.shape)
            semantic_labels_maxed = semantic_labels * 1000

            # 3. get the instances IDs
            # print(semantic_pred.shape)
            # print(center_pred.shape)
            # print(x_offset_pred.shape)
            # print(y_offset_pred.shape)
            # shapes:
            # torch.Size([1, 21, 513, 513])
            # torch.Size([1, 1, 513, 513])
            # torch.Size([1, 1, 513, 513])
            # torch.Size([1, 1, 513, 513])
            # expected shapes
            # torch.Size([1, 513, 513])
            # torch.Size([1, 513, 513])
            # torch.Size([1, 513, 513])
            # torch.Size([1, 513, 513])
            center_pred = center_pred[0]
            x_offset_pred = x_offset_pred[0]
            y_offset_pred = y_offset_pred[0]
            instances = self.get_instances(
                semantic_labels, center_pred, x_offset_pred, y_offset_pred
            )

            # 4. and add to semantic_labels
            final_instance_image = semantic_labels_maxed.cuda() + instances
            # print("final_instance_image.shape: ", final_instance_image.shape)

            # frankfurt_000000_000294_gtFine_labelIds.png ->
            # with frankfurt_000000_000294_gtFine_instanceIds.png
            save_filepath = (
                directorypath
                + "/"
                + filename.replace("labelIds", "instanceIds")
            )
            # print(save_filepath)
            # exit(0)
            # instance_image = np.array(Image.open(filepath[0]))
            # plt.imshow(instance_image)
            # plt.show()

            # save instance image (finally!)
            print(save_filepath)
            final_instance_image = np.uint8(final_instance_image[0].cpu().numpy())
            instance_image = Image.fromarray(final_instance_image)
            instance_image.save(save_filepath)

    def get_instances(
        self, semantic_labels, center, x_offset, y_offset, center_threshold=250
    ):
        mask = torch.zeros_like(semantic_labels)
        for num in self.things_category:
            mask = torch.where(semantic_labels == num, semantic_labels, mask)

        # remove pixels in center that belong to stuffs
        center = center * mask.cuda()

        # 2.0 get center points
        # max pool according to paper
        centers_max = F.max_pool2d(center, 7, stride=1, padding=3)
        centers_select = torch.where(
            center == centers_max, center, torch.zeros_like(center)
        )

        # 2.1 choose top k points, need to sort, then choose the top
        centers_select = torch.where(
            centers_select > center_threshold,
            torch.ones_like(centers_select) * 255,
            torch.zeros_like(centers_select),
        )

        # 3.0 do regressions
        # first make tensors that represent where the centers are for each point
        _, h, w = x_offset.shape
        gridy, gridx = torch.meshgrid(torch.arange(h), torch.arange(w))
        gridx = gridx.cuda().unsqueeze(0)
        gridy = gridy.cuda().unsqueeze(0)

        offsetted_pixloc_x = gridx + x_offset
        offsetted_pixloc_y = gridy + y_offset

        # get indices of center points TODO: ensure correct axis
        center_points = centers_select.nonzero()
        if center_points.shape[0] < 1:
            return torch.zeros_like(semantic_labels[0]).cuda()
        center_points_x = center_points[:, 2:3].unsqueeze(
            -1
        ) * torch.ones_like(x_offset)
        center_points_y = center_points[:, 1:2].unsqueeze(
            -1
        ) * torch.ones_like(x_offset)

        distance_x = (center_points_x - offsetted_pixloc_x) ** 2
        distance_y = (center_points_y - offsetted_pixloc_y) ** 2

        distance_xy = torch.sqrt(distance_x + distance_y)

        instances = (torch.argmin(distance_xy, 0) + 1).float()
        instances = torch.where(
            mask.cuda() == 0, torch.zeros_like(instances), instances,
        )

        # for debug
        # show_image = centers_select[0].cpu().numpy()
        # plt.imshow(show_image)
        # plt.show()
        # exit(0)
        return instances

    def test_grouping(self, epoch):
        self.model.eval()
        # self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, label, center, x_reg, y_reg = (
                sample["image"],
                sample["label"],
                sample["center"],
                sample["x_reg"],
                sample["y_reg"],
            )

            # assume groundtruth is prediction (remove multiplier in final)
            if self.args.cuda:
                image, label, center, x_reg, y_reg = (
                    image.cuda(),
                    label.cuda(),
                    center.cuda(),
                    x_reg.cuda() / 2.0,
                    y_reg.cuda() / 2.0,
                )
            # print(label.shape)
            # print(center.shape)
            # print(x_reg.shape)
            # print(y_reg.shape)
            # exit(0)
            # preprocess ~ return to original size.
            # works only for test_loader
            # prediction = F.interpolate(
            #     label_pred,
            #     size=image.size()[2:],
            #     mode="bilinear",
            #     align_corners=True,
            # )

            # 1.0 filter out stuff categories
            # 1.1 Using categories list
            # self.things_category = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 30]
            mask = torch.zeros_like(label)
            for num in self.things_category:
                mask = torch.where(label == num, label, mask)

            # 1.2 using mask from y regression
            # mask = torch.where(
            #     y_reg * y_reg == 0,
            #     torch.zeros_like(label),
            #     torch.ones_like(label),
            # )

            # instance_label = torch.where(
            #     y_reg * y_reg == 0, torch.zeros_like(label), label
            # )

            # 2.0 get center points
            # max pool according to paper
            centers_max = F.max_pool2d(center, 7, stride=1, padding=3)
            centers_select = torch.where(
                center == centers_max, center, torch.zeros_like(center)
            )
            # 2.1 choose top k points, need to sort, then choose the top
            # use thresholding=2.5 (of 255)
            centers_select = torch.where(
                centers_select > 250,
                torch.ones_like(centers_select) * 255,
                torch.zeros_like(centers_select),
            )

            # 3.0 do regressions
            # first make tensors that represent where the centers are for each point
            _, h, w = x_reg.shape
            gridy, gridx = torch.meshgrid(torch.arange(h), torch.arange(w))
            gridx = gridx.cuda().unsqueeze(0)  # + x_reg
            gridy = gridy.cuda().unsqueeze(0)  # + y_reg

            offsetted_pixloc_x = gridx + x_reg
            offsetted_pixloc_y = gridy + y_reg

            # get indices of center points TODO: ensure correct axis
            center_points = centers_select.nonzero()
            if center_points.shape[0] < 1:
                continue
            center_points_x = center_points[:, 2:3].unsqueeze(
                -1
            ) * torch.ones_like(x_reg)
            center_points_y = center_points[:, 1:2].unsqueeze(
                -1
            ) * torch.ones_like(x_reg)

            distance_x = (center_points_x - offsetted_pixloc_x) ** 2
            distance_y = (center_points_y - offsetted_pixloc_y) ** 2

            distance_xy = torch.sqrt(distance_x + distance_y)

            group_ids = (torch.argmin(distance_xy, 0) + 1).float()
            group_ids = torch.where(
                mask[0] == 0, torch.zeros_like(group_ids), group_ids,
            )

            show_image = group_ids.cpu().numpy()

            # to get final instance ID, multiply groupID and classID

            # display outputs
            semantic_show = decode_seg_map_sequence(
                label.cpu().numpy(), dataset=self.args.dataset,
            )[0].permute(1, 2, 0)

            # out centers
            # centers_show = centers_select[0].data.cpu().numpy()
            centers_show = center[0].data.cpu().numpy()

            # out x_reg prediction
            # out_image = centers_reg[0][0].data.cpu().numpy()

            # out y_reg prediction
            # out_image = centers_reg[0][0].data.cpu().numpy()

            # # out x_reg gt
            # out_image = x_reg[0].data.cpu().numpy()

            # # out y_reg gt
            # out_image = y_reg[0].data.cpu().numpy()

            # show image
            image = image[0].permute(1, 2, 0).cpu().numpy()
            image *= (0.229, 0.224, 0.225)
            image += (0.485, 0.456, 0.406)
            image *= 255.0
            image = image.astype(np.uint8)

            plt.figure()
            plt.subplot(221)
            plt.imshow(image)
            plt.subplot(222)
            plt.imshow(semantic_show)
            plt.subplot(223)
            plt.imshow(centers_show)
            plt.subplot(224)
            plt.imshow(show_image)
            plt.show()

    def evaluate(self, epoch):
        self.model.eval()
        # self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc="\r")
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, label = sample["image"], sample["label"]
            if self.args.cuda:
                image, label = image.cuda(), label.cuda()
                input_image = F.interpolate(
                    image,
                    size=[513, 513],
                    mode="bilinear",
                    align_corners=True,
                )
            with torch.no_grad():
                try:
                    output = self.model(input_image)
                except ValueError as identifier:
                    # there was an error with wrong input size
                    print("Error: ", identifier)
                    continue
            prediction = F.interpolate(
                output[0],
                size=image.size()[2:],
                mode="bilinear",
                align_corners=True,
            )
            prediction = prediction.data.cpu().numpy()
            # label = label.cpu().numpy()
            prediction = np.argmax(prediction, axis=1)
            centers = output[1]
            plt.imshow(centers[0][0].data.cpu().numpy())

            centers = F.interpolate(
                centers,
                size=image.size()[2:],
                mode="bilinear",
                align_corners=True,
            )

            # max pool according to paper
            centers_new = F.max_pool2d(centers, 7, stride=1, padding=3)

            # use thresholding=0.1 (of 255)
            centers = torch.where(
                centers == centers_new, centers, torch.zeros_like(centers)
            )
            # choose top k points, need to sort, then choose the top
            # centers = torch.where(centers>10, torch.ones_like(centers), torch.zeros_like(centers))
            centers_reg = F.interpolate(
                output[2],
                size=image.size()[2:],
                mode="bilinear",
                align_corners=True,
            )

            # print(output[2].shape)
            # exit(0)
            points = (centers > 0.1 * 255).nonzero()
            print("points.shape: ", points.shape)
            print("centers_reg.shape: ", centers_reg.shape)
            points_only = points[:, 2:].unsqueeze(-1).unsqueeze(
                -1
            ) * torch.ones_like(centers_reg)
            print("points_only.shape: ", points_only.shape)

            x_reg = centers_reg[0][0]
            y_reg = centers_reg[0][1]
            print("x_reg.shape: ", x_reg.shape)
            print("prediction.shape: ", prediction.shape)

            diff_sqr = (points_only - centers_reg) ** 2
            mag = torch.sqrt(diff_sqr[:, 0, :, :] + diff_sqr[:, 1, :, :])
            print("mag: ", mag.shape)
            final = torch.argmin(mag, 0)
            # print(final)
            # show = (final/torch.max(final)).data.cpu().numpy()
            # plt.imshow(final.data.cpu().numpy())
            # plt.show()
            # h, w = x_reg.shape
            # gridx, gridy = torch.meshgrid(torch.arange(h), torch.arange(w))
            # x_new = gridx.cuda() + x_reg
            # y_new = gridy.cuda() + y_reg
            # exit(0)

            # # display outputs
            # out_image = decode_seg_map_sequence(
            #     prediction, dataset=self.args.dataset,
            # )[0].permute(1, 2, 0)

            out_image = centers_reg[0][0].data.cpu().numpy()

            img_tmp = np.transpose(image[0].cpu().numpy(), axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)

            plt.figure()
            plt.subplot(121)
            plt.imshow(img_tmp)
            # this should be final image
            plt.subplot(122)
            plt.imshow(out_image)
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Panoptic Deeplab Training"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet",
        choices=["xception_3stage", "mobilenet_3stage", "resnet_3stage"],
        help="backbone name (default: resnet)",
    )
    parser.add_argument(
        "--out-stride",
        type=int,
        default=16,
        help="network output stride (default: 8)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pascal",
        choices=["pascal", "coco", "cityscapes"],
        help="dataset name (default: pascal)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="segmentation",
        choices=["segmentation", "panoptic"],
        help="training task (default: segmentation)",
    )
    parser.add_argument(
        "--use-sbd",
        action="store_true",
        default=True,
        help="whether to use SBD dataset (default: True)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="dataloader threads",
    )
    parser.add_argument(
        "--base-size", type=int, default=513, help="base image size"
    )
    parser.add_argument(
        "--crop-size", type=int, default=513, help="crop image size"
    )
    parser.add_argument(
        "--sync-bn",
        type=bool,
        default=None,
        help="whether to use sync bn (default: auto)",
    )
    parser.add_argument(
        "--freeze-bn",
        type=bool,
        default=False,
        help="whether to freeze bn parameters (default: False)",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="loss func type (default: ce)",
    )
    # training hyper params
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="number of epochs to train (default: auto)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        metavar="N",
        help="start epochs (default:0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="input batch size for \
                                training (default: auto)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="input batch size for \
                                testing (default: auto)",
    )
    parser.add_argument(
        "--use-balanced-weights",
        action="store_true",
        default=False,
        help="whether to use balanced weights (default: False)",
    )
    # optimizer params
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (default: auto)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="poly",
        choices=["poly", "step", "cos"],
        help="lr scheduler mode: (default: poly)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        metavar="M",
        help="w-decay (default: 5e-4)",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=False,
        help="whether use nesterov (default: False)",
    )
    # cuda, seed and logging
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    # checking point
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="put the path to resuming file if needed",
    )
    parser.add_argument(
        "--checkname", type=str, default=None, help="set the checkpoint name"
    )
    # finetuning pre-trained models
    parser.add_argument(
        "--ft",
        action="store_true",
        default=False,
        help="finetuning on a different dataset",
    )
    # evaluation option
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="evaluation interval (default: 1)",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        default=False,
        help="skip validation during training",
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        except ValueError:
            raise ValueError(
                "Argument --gpu_ids must be a comma-separated list of integers only"
            )

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.test_batch_size is None:
        args.test_batch_size = 1

    print(args)
    torch.manual_seed(args.seed)
    tester = Tester(args)
    # tester.evaluate(0)
    # tester.test_grouping(0)
    tester.test_saving(0)


if __name__ == "__main__":
    main()
