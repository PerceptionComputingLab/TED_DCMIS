c# Three dataset for continual segmentation

## * Diffusion model generation
    python generatation_DM

**important hyper-paramters**

    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--img-size', type=int, default=96, help='image size')
    parser.add_argument('--timesteps', type=int, default=300, help='time steps')
    parser.add_argument('--epochs', type=int, default=300, help='# of epochs')
    parser.add_argument('--interval', type=int, default=100, help='# of interval')
    parser.add_argument('--inference_n', type=int, default=100, help='# of inference')

**important pre-trained weight** [download](https://pan.baidu.com/s/1hxHN9Iq8GvBbz_Z3pXBBcg?pwd=60ir)

## Prostate segmentation (6 domains)
[download](https://drive.google.com/file/d/1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-/view)
1. intensity is rescaled to [0., 1.]
2. rename dataset BMC from 'Seg' to 'seg'
3. relabel to binary in dataset 'RUNMC', 'BMC'




    python prostate_prepare.py

## cardiac segmentation, respectively LV-endo, LV-epi, RV (5 domains)
[ACDC](https://aistudio.baidu.com/aistudio/datasetdetail/56020)

1. intensity is rescaled to [0., 1.]
2. split into 5 group based on information in file Info.cfg
3. add specific noisy in each group