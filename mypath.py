class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            # folder that contains VOCdevkit/.
            return '/home/tjosh/datasets/VOCdevkit/VOC2012/'
        elif dataset == 'sbd':
            # folder that contains dataset/.
            return '/home/tjosh/datasets/semseg_benchmark/benchmark_RELEASE/'
        elif dataset == 'cityscapes':
            # foler that contains leftImg8bit/
            return '/path/to/datasets/cityscapes/'
        elif dataset == 'coco':
            return '/media/tjosh/vault/dataset/panoptic_segmentation/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
