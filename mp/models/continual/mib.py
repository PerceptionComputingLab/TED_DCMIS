from mp.models.model import Model


class MIB(Model):
    r"""Knowledge Distillation as porposed in Incremental learning techniques for semantic segmentation
    by Michieli, U., Zanuttigh, P., 2019
    """

    def __init__(
        self,
        input_shape=(1, 256, 256),
        nr_labels=2,
        backbone="unet",
        unet_dropout=0,
        unet_monte_carlo_dropout=0,
        unet_preactivation=False,
    ):
        r"""Constructor

        Args:
            input_shape (tuple of int): input shape of the images
            nr_labels (int): number of labels for the segmentation
            unet_dropout (float): dropout probability for the U-Net
            unet_monte_carlo_dropout (float): monte carlo dropout probability for the U-Net
            unet_preactivation (boolean): whether to use U-Net pre-activations
        """
        super(MIB, self).__init__()

        self.input_shape = input_shape
        self.nr_labels = nr_labels
        self.backbone = backbone

        self.unet_dropout = unet_dropout
        self.unet_monte_carlo_dropout = unet_monte_carlo_dropout
        self.unet_preactivation = unet_preactivation

        self.init_backbone()
