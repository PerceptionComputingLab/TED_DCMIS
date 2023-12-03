import argparse
import torch


# parse train options
def _get_parser():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--experiment-name', type=str, default='', help='experiment name for new or resume')

    # hardware
    parser.add_argument('--device', type=str, default='cuda', help='device type cpu or cuda')
    parser.add_argument("--device-ids", nargs="+", default=[5], type=int, help="ID(s) of GPU device(s)")
    parser.add_argument('--n-workers', type=int, default=2, help='# multiplied by # of GPU to get # of total workers')

    # dataset
    parser.add_argument('--dataset', type=str, default='mm', help='datasets to training')
    parser.add_argument('--target_class', type=str, default='i', help='cardiac target. i:LV-endo, o:LV-epi, r:RV')
    parser.add_argument('--test-ratio', type=float, default=0.25, help='ratio of data to be used for testing')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='ratio of data to be used for validation')
    parser.add_argument('--input_dim_c', type=int, default=1, help='input channels for images')
    parser.add_argument('--input_dim_hw', type=int, default=192, help='height and width for images')
    parser.add_argument('--no-resize', action='store_true', help='specify if images should not be resized')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='# of samples per dataloader, only use when debugging')

    # training
    parser.add_argument('--epochs', type=int, default=1, help='# of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for initial training stage')
    parser.add_argument('--lr-2', type=float, default=1e-4, help='learning rate for continual training stage')
    parser.add_argument('--approach', type=str, default='joint', help='which approach to train')
    parser.add_argument('--backbone', type=str, default='unet', help='backbone:unet or swinunet')
    parser.add_argument('--val-best', action='store_true', default=True, help='best validation to next domain')
    parser.add_argument('--gugf', action='store_true', default=False, help='use gugf')

    # resume training
    parser.add_argument('--resume-epoch', type=int, default=None,
                        help='resume training at epoch, -1 for latest, select run using experiment-name argument')

    # logging
    parser.add_argument('--eval-interval', type=int, default=7, help='evaluation interval (on all datasets)')
    parser.add_argument('--save-interval', type=int, default=1, help='save interval')
    parser.add_argument('--display-interval', type=int, default=1, help='display/tensorboard interval')
    parser.add_argument('--run-loss-print-interval', type=int, default=1, help='run_loss_print_interval')

    # loss weighting
    parser.add_argument('--loss-type', type=str, default='dice_bce', help='which type of loss to train')
    parser.add_argument('--lambda-d', type=float, default=0.001, help='lambda for tuning MAS or KD loss')

    # noise level on ska
    parser.add_argument('--mean', type=float, default=0, help='lambda for tuning mean')
    parser.add_argument('--std', type=float, default=1, help='lambda for tuning std')

    # U-Net
    parser.add_argument('--unet-dropout', type=float, default=0, help='apply dropout to UNet')
    parser.add_argument('--unet-monte-carlo-dropout', type=float, default=0, help='apply monte carlo dropout to UNet')
    parser.add_argument('--unet-preactivation', action='store_true',
                        help='UNet preactivation; True: norm, act, conv; False:conv, norm, act')

    return parser


def parse_args(argv):
    """Parses arguments passed from the console as, e.g.
    'python ptt/main.py --epochs 3' """

    parser = _get_parser()
    args = parser.parse_args(argv)

    args.device = str(
        args.device + ':' + str(args.device_ids[0]) if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    device_name = str(torch.cuda.get_device_name(args.device) if args.device == "cuda" else args.device)
    print('Device name: {}'.format(device_name))
    args.input_shape = (args.input_dim_c, args.input_dim_hw, args.input_dim_hw)

    return args


def parse_args_as_dict(argv):
    """Parses arguments passed from the console and returns a dictionary """
    return vars(parse_args(argv))


def parse_dict_as_args(dictionary):
    """Parses arguments given in a dictionary form"""
    argv = []
    for key, value in dictionary.items():
        if isinstance(value, bool):
            if value:
                argv.append('--' + key)
        else:
            argv.append('--' + key)
            argv.append(str(value))
    return parse_args(argv)
