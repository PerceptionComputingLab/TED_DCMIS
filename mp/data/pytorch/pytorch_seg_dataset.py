import copy
from mp.data.pytorch.pytorch_dataset import PytorchDataset
import mp.data.pytorch.transformation as trans
import mp.eval.inference.predictor as pred


class PytorchSegmnetationDataset(PytorchDataset):
    def __init__(
        self,
        dataset,
        ix_lst=None,
        size=None,
        norm_key="rescaling",
        aug_key="standard",
        channel_labels=True,
    ):
        r"""A torch.utils.data.Dataset for segmentation data.
        Args:
            dataset (SegmentationDataset): a SegmentationDataset
            ix_lst (list[int)]): list specifying the instances of the dataset.
                If 'None', all not in the hold-out dataset are incuded.
            size (tuple[int]): size as (channels, width, height, Opt(depth))
            norm_key (str): Normalization strategy, from
                mp.data.pytorch.transformation
            aug_key (str): Augmentation strategy, from
                mp.data.pytorch.transformation
            channel_labels (bool): if True, the output has one channel per label
        """
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size)
        self.norm = trans.NORMALIZATION_STRATEGIES[norm_key]
        self.aug = trans.AUGMENTATION_STRATEGIES[aug_key]
        self.nr_labels = dataset.nr_labels
        self.channel_labels = channel_labels
        self.predictor = None

    def get_instance(self, ix=None, name=None):
        r"""Get a particular instance from the ix or name"""
        assert ix is None or name is None
        if ix is None:
            instance = [ex for ex in self.instances if ex.name == name]
            assert len(instance) == 1
            return instance[0]
        else:
            return self.instances[ix]

    def get_ix_from_name(self, name):
        r"""Get ix from name"""
        return next(ix for ix, ex in enumerate(self.instances) if ex.name == name)

    def transform_subject(self, subject):
        r"""Transform a subject by applying normalization and augmentation ops"""
        if self.norm is not None:
            subject = self.norm(subject)
        if self.aug is not None:
            subject = self.aug(subject)
        return subject

    def get_subject_dataloader(self, subject_ix):
        r"""Get a list of input/target pairs equivalent to those if the dataset
        was only of subject with index subject_ix. For evaluation purposes.
        """
        raise NotImplementedError


class PytorchSeg2DDataset(PytorchSegmnetationDataset):
    r"""Divides images into 2D slices. If resize=True, the slices are resized to
    the specified size, otherwise they are center-cropped and padded if needed.
    """

    def __init__(
        self,
        dataset,
        ix_lst=None,
        size=(1, 256, 256),
        norm_key="rescaling",
        aug_key="standard",
        channel_labels=True,
        resize=False,
    ):
        if isinstance(size, int):
            size = (1, size, size)
        super().__init__(
            dataset=dataset,
            ix_lst=ix_lst,
            size=size,
            norm_key=norm_key,
            aug_key=aug_key,
            channel_labels=channel_labels,
        )
        assert len(self.size) == 3, "Size should be 2D"
        self.resize = resize
        self.predictor = pred.Predictor2D(
            self.instances, size=self.size, norm=self.norm, resize=resize
        )

        self.idxs = []
        for instance_ix, instance in enumerate(self.instances):
            for slide_ix in range(instance.shape[-1]):
                self.idxs.append((instance_ix, slide_ix))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        r"""Returns x and y values each with shape (c, w, h)"""
        instance_idx, slice_idx = self.idxs[idx]

        # print('reading', instance_idx, 'using copy.deepcopy and multi workers')
        subject = copy.deepcopy(self.instances[instance_idx].get_subject())
        subject.load()

        # None aug, None rescale. Data has been scaled to [0., 1.]
        # subject = self.transform_subject(subject)

        x = subject.x.tensor.permute(3, 0, 1, 2)[slice_idx].float()
        y = subject.y.tensor.permute(3, 0, 1, 2)[slice_idx].float()

        if self.resize:
            x = trans.resize_2d(x, size=self.size)
            y = trans.resize_2d(y, size=self.size, label=True)
        else:
            x = trans.centre_crop_pad_2d(x, size=self.size)
            y = trans.centre_crop_pad_2d(y, size=self.size)

        if self.channel_labels:
            y = trans.per_label_channel(y, self.nr_labels)

        return x, y

    def get_subject_dataloader(self, subject_ix):
        dl_items = []
        idxs = [
            idx
            for idx, (instance_idx, slice_idx) in enumerate(self.idxs)
            if instance_idx == subject_ix
        ]
        for idx in idxs:
            x, y = self.__getitem__(idx)
            dl_items.append((x.unsqueeze_(0), y.unsqueeze_(0)))
        return dl_items
