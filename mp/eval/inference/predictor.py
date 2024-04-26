# ------------------------------------------------------------------------------
# A Predictor makes a prediction for a subject index that has the same size
# as the subject's target. It reverses the trandormation operations performed
# so that inputs can be passed through the model. It, for instance, merges
# patches and 2D slices into 3D volumes of the original size.
# ------------------------------------------------------------------------------

import copy
import torch
import torchio
import mp.data.pytorch.transformation as trans


class Predictor:
    r"""A predictor recreates a prediction with the correct dimensions from
    model outputs. There  different predictors for different PytorchDatasets,
    and these are setted internally with the creation of a PytorchDataset.
    Args:
        instances (list[Instance]): a list of instances, as for a Dataset
        size (tuple[int]): size as (channels, width, height, Opt(depth))
        norm (torchio.transforms): a normaliztion strategy
    """

    def __init__(self, instances, size=(1, 56, 56, 10), norm=None):
        self.instances = instances
        assert len(size) > 2
        self.size = size
        self.norm = norm

    def transform_subject(self, subject):
        r"""Apply normalization strategy to subject."""
        if self.norm is not None:
            subject = self.norm(subject)
        return subject

    def get_subject(self, subject_ix):
        r"""Copy and load a TorchIO subject."""
        subject = copy.deepcopy(self.instances[subject_ix].get_subject())
        subject.load()
        subject = self.transform_subject(subject)
        return subject

    def get_subject_prediction(self, agent, subject_ix):
        r"""Get a prediction for a 3D subject."""
        raise NotImplementedError


class Predictor2D(Predictor):
    r"""The Predictor2D makes a forward pass for each 2D slice and merged these
    into a volume.
    """

    def __init__(self, *args, resize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize = resize

    def get_subject_prediction(self, agent, subject_ix, probabilities=False):

        subject = self.get_subject(subject_ix)

        # Slides first
        x = subject.x.tensor.permute(3, 0, 1, 2)
        # Get original size
        original_size = subject["y"].data.shape
        original_size_2d = original_size[:3]

        pred = []
        if probabilities:
            pred_prob = []
        with torch.no_grad():
            for slice_idx in range(len(x)):
                if self.resize:
                    inputs = trans.resize_2d(x[slice_idx], size=self.size).to(
                        agent.device
                    )
                    inputs = torch.unsqueeze(inputs, 0)
                    if probabilities:
                        slice_pred_prob = agent.get_outputs(inputs).float()
                        slice_pred = torch.argmax(slice_pred_prob, dim=1)
                        slice_pred_prob = slice_pred_prob[:, 0]
                        pred_prob.append(slice_pred_prob)
                    else:
                        slice_pred = agent.predict(inputs).float()
                    pred.append(
                        trans.resize_2d(slice_pred, size=original_size_2d, label=True)
                    )
                else:
                    inputs = trans.centre_crop_pad_2d(x[slice_idx], size=self.size).to(
                        agent.device
                    )
                    inputs = torch.unsqueeze(inputs, 0)
                    if probabilities:
                        slice_pred_prob = agent.get_outputs(inputs).float()
                        slice_pred = torch.argmax(slice_pred_prob, dim=1)
                        slice_pred_prob = slice_pred_prob[:, 0]
                        pred_prob.append(slice_pred_prob)
                    else:
                        slice_pred = agent.predict(inputs).float()
                    pred.append(
                        trans.centre_crop_pad_2d(slice_pred, size=original_size_2d)
                    )

        # Merge slices and rotate so depth last
        pred = torch.stack(pred, dim=0)
        pred = pred.permute(1, 2, 3, 0)
        if probabilities:
            pred_prob = torch.stack(pred_prob, dim=0)
            pred_prob = pred_prob.permute(1, 2, 3, 0)
            assert original_size == pred_prob.shape
            return pred, pred_prob
        assert original_size == pred.shape
        return pred
