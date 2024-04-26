import os

import torch
import argparse

from torch.utils.data import DataLoader
from data_prep.diffusion_utils.dataset_diffusion import PredictionDataset
from data_prep.diffusion_utils.diffusion_model_2D import SimpleUnet
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# diffusion model path
diffusion_log_path = os.path.join(
    "/home/zhuzhanshi/MedicalCL/storage/data", "PseudoData"
)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(
        device
    ) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def sample_timestep(x, t, model):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def sample_plot_image(out_name, model, device, rgb=True):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)

    plt.figure(figsize=(15, 15))
    plt.axis("off")

    num_images = 5
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i // stepsize + 1)
            show_tensor_image(img.detach().cpu(), out_name, rgb)
    if rgb:
        plt.savefig(out_name)
        # plt.show()


def show_tensor_image(image, out_name, rgb=True):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: t.clamp(-1, 1)),
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    image = reverse_transforms(image)
    image = np.array(image)

    if rgb:
        plt.imshow(image)
    else:
        cv2.imwrite(out_name, image)


def show_images(
    data,
    out_name,
    num_samples=10,
    cols=5,
):
    """Plots some samples from the dataset"""
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(num_samples // cols + 1, cols, i + 1)
        plt.imshow(img[0])
    plt.savefig(out_name)
    # plt.show()


def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


def train(sub):
    data = PredictionDataset(subset=sub, size=(args.img_size, args.img_size))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    show_images(data, os.path.join(log_path, "samples.png"))

    model = SimpleUnet()
    model = torch.load("model_2d.pkl")  # load well-pretrained weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            pbar.update()
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
                loss = get_loss(model, batch, t, device)
                loss.backward()
                optimizer.step()

                if epoch % args.interval == 0 and step == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                    log_name = os.path.join(log_path, "train", str(epoch) + ".png")
                    sample_plot_image(log_name, model, device, rgb=True)

    torch.save(model, os.path.join(log_path, "train", sub + "_2d.pkl"))


def inference(sub):
    model = torch.load(os.path.join(log_path, "train", sub + "_2d.pkl"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    num_samples = args.inference_n
    with tqdm(total=num_samples) as pbar:
        for i in range(num_samples):
            pbar.update()
            log_name = os.path.join(log_path, "inference", str(i).zfill(4) + ".png")
            sample_plot_image(log_name, model, device, rgb=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", type=str, default=["MINF"], help="which dataset to generate"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--img-size", type=int, default=96, help="image size")
    parser.add_argument("--timesteps", type=int, default=300, help="time steps")
    parser.add_argument("--epochs", type=int, default=300, help="# of epochs")
    parser.add_argument("--interval", type=int, default=100, help="# of interval")
    parser.add_argument("--inference_n", type=int, default=100, help="# of inference")

    args = parser.parse_args()

    # datasets = args.datasets
    # datasets = ['RUNMC', 'BMC', 'I2CVB', 'UCL', 'BIDMC', 'HK',
    #             'HCM', 'DCM', 'NOR', 'MINF', 'RV']
    datasets = ["Philips", "GE", "Canon", "Siemens"]
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.img_size
    epochs = args.epochs  # Try more!

    # Define beta schedule
    T = args.timesteps
    betas = linear_beta_schedule(timesteps=T)

    # Pre-calculate different terms for closed form
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    for subset in datasets:
        print("start", subset)
        log_path = os.path.join(diffusion_log_path, subset)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            os.makedirs(os.path.join(log_path, "train"))
            os.makedirs(os.path.join(log_path, "inference"))
        train(subset)
        inference(subset)

    print("Done!!!", datasets)
