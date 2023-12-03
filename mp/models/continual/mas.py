from mp.models.model import Model
from mp.models.segmentation.unet_fepegar import UNet2D
import torch.optim as optim


class MAS(Model):
    r"""Memory Aware Synapses for brain segmentation
    as porposed in Importance driven continual learning for segmentation across domains by Oezguen et al., 2020
    """

    def __init__(self,
                 input_shape=(1, 256, 256),
                 nr_labels=2,
                 backbone='unet',
                 unet_dropout=0,
                 unet_monte_carlo_dropout=0,
                 unet_preactivation=False
                 ):
        r"""Constructor
        
        Args:
            input_shape (tuple of int): input shape of the images
            nr_labels (int): number of labels for the segmentation
            unet_dropout (float): dropout probability for the U-Net
            unet_monte_carlo_dropout (float): monte carlo dropout probability for the U-Net
            unet_preactivation (boolean): whether to use U-Net pre-activations
        """
        super(MAS, self).__init__()

        self.input_shape = input_shape
        self.nr_labels = nr_labels
        self.backbone = backbone

        self.unet_dropout = unet_dropout
        self.unet_monte_carlo_dropout = unet_monte_carlo_dropout
        self.unet_preactivation = unet_preactivation

        self.init_backbone()

        self.importance_weights = None
        self.tasks = 0

        self.n_params_unet = sum(p.numel() for p in self.unet_new.parameters())

    def update_importance_weights(self, importance_weights):
        r"""Update importance weights w/ computed ones

        Args:
            (torch.Tensor or list): importance_weights
        """
        if self.importance_weights == None:
            self.importance_weights = importance_weights
        else:
            for i in range(len(self.importance_weights)):
                self.importance_weights[i] -= self.importance_weights[i] / self.tasks
                self.importance_weights[i] += importance_weights[i] / self.tasks
        self.tasks += 1
