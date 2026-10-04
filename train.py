import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.utils import *
from utils.models import *
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser=argparse.ArgumentParser()

    parser.add_argument('--content_dir',type=str,default='C:\\Users\\Harsh\\OneDrive\\Documents\\Desktop\\NST_CODE\\content_data',help='Location of Content dataset')
    parser.add_argument('--style_dir',type=str,default='C:\\Users\\Harsh\\OneDrive\\Documents\\Desktop\\NST_CODE\\style_data',help='Location of Style dataset')
    parser.add_argument('--vgg',type=str,default='C:\\Users\\Harsh\\OneDrive\\Documents\\Desktop\\NST_CODE\\vgg_normalised.pth',help='Location of VGG19 model')
    parser.add_argument('--experiment',type=str,default='big_dataset',help='Experiment name (outputs go to ./experiments/<name>)')
    parser.add_argument('--crop',action='store_true',default=True,help='Crop the image to the specified size')
    parser.add_argument('--batch_size',type=int,default=4,help='Per-step batch size (kept fixed, raise accum_steps instead)')
    parser.add_argument('--lr',type=float,default=1e-4,help='Learning rate for training')
    parser.add_argument('--lr_decay',type=float,default=5e-5,help='Learning rate decay for training')
    parser.add_argument('--epochs',type=int,default=10,help='Total number of epochs (end epoch)')
    parser.add_argument('--start_epoch',type=int,default=0,help='Epoch to continue from when resuming (0 = fresh start)')
    parser.add_argument('--content_weight',type=float,default=1.0,help='Weight for content loss')

    # ---- per-epoch updates ----
    parser.add_argument('--style_start',type=float,default=1.0,help='Style weight at the first epoch')
    parser.add_argument('--style_max',type=float,default=15.0,help='Style weight at the last epoch (linear ramp)')
    parser.add_argument('--size_schedule',type=str,default='256,256,256,256,256,256,384,384,512,512',
                        help='Final (crop) image size for each epoch, comma-separated')
    parser.add_argument('--accum_schedule',type=str,default='1,1,1,2,2,2,3,3,4,4',
                        help='accum_steps for each epoch, comma-separated (effective batch = batch_size * accum_steps)')
    parser.add_argument('--resize_ratio',type=float,default=1.5,help='Resize size before cropping = final_size * ratio')

    parser.add_argument('--log_interval', type=int, default=1, help='Epochs between logging')
    parser.add_argument('--save_interval', type=int, default=1, help='Epochs between saving model checkpoints and output image')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from a checkpoint')
    parser.add_argument('--decoder_pth', type=str, default=None, help='Path to the decoder checkpoint')
    parser.add_argument('--optimizer_pth', type=str, default=None, help='Path to the optimizer checkpoint')

    return parser.parse_args()


def pick(values, epoch):
    # value for this epoch; the last value is reused if the list is shorter than the epoch count
    return values[min(epoch, len(values) - 1)]


def make_loaders(args, final_size):
    resize_size = int(final_size * args.resize_ratio)
    content_transform = get_transform(resize_size, args.crop, final_size)
    style_transform = get_transform(resize_size, args.crop, final_size)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return content_dataloader, style_dataloader


def main():
    args = parse_arguments()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_dir = Path(f'./experiments/{args.experiment}')
    save_dir.mkdir(parents=True, exist_ok=True)

    #Save Arguments values
    with open(f'{save_dir}/args.txt', 'a') as args_files:
        args_files.write(f'--- run starting at epoch {args.start_epoch} ---\n')
        for key,value in vars(args).items():
            args_files.write(f'{key}: {value}\n')

    size_schedule = [int(x) for x in args.size_schedule.split(',')]
    accum_schedule = [int(x) for x in args.accum_schedule.split(',')]

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = VGGDecoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0/(1.0+args.lr_decay*epoch))

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_pth))
        optimizer.load_state_dict(torch.load(args.optimizer_pth))
        for g in optimizer.param_groups:      # apply the lr given on the command line
            g['lr'] = args.lr

    mse_loss = torch.nn.MSELoss()

    encoder.eval()       # frozen, feature extractor only
    decoder.train()      # decoder is what we're training

    current_size = None
    print("Training...")

    for epoch in range(args.start_epoch, args.epochs):

        # ---- per-epoch updates ----
        # 1) style weight: linear ramp style_start -> style_max over the whole run
        args.style_weight = args.style_start + (args.style_max - args.style_start) * epoch / max(1, args.epochs - 1)
        # 2) image size
        final_size = pick(size_schedule, epoch)
        # 3) accumulation steps (effective batch size)
        accum_steps = pick(accum_schedule, epoch)

        # rebuild the dataloaders only when the image size changes
        if final_size != current_size:
            content_dataloader, style_dataloader = make_loaders(args, final_size)
            current_size = final_size
            print(f'Number of batches in content dataset: {len(content_dataloader)}')
            print(f'Number of batches in style dataset: {len(style_dataloader)}')

        tqdm.write(
            f'Epoch {epoch+1}/{args.epochs}: style_weight={args.style_weight:g}, '
            f'image_size={final_size}, accum_steps={accum_steps} '
            f'(effective batch={args.batch_size * accum_steps})'
        )

        total_steps = min(len(content_dataloader), len(style_dataloader))
        progress_bar = tqdm(
            enumerate(zip(content_dataloader, style_dataloader)),
            total=total_steps,
            desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch'
        )

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0

        optimizer.zero_grad()

        for i, (content_batch, style_batch) in progress_bar:
            content_images = content_batch.to(device)
            style_images = style_batch.to(device)

            with torch.no_grad():
                content_features = encoder(content_images)
                style_features = encoder(style_images)
                t = adaIN(content_features[-1], style_features[-1])

            generated_images = decoder(t)
            generated_features = encoder(generated_images)

            content_loss = mse_loss(generated_features[-1], t) * args.content_weight

            style_loss = 0.0
            for gf, sf in zip(generated_features, style_features):
                gf_mean, gf_std = calc_mean_std(gf)
                sf_mean, sf_std = calc_mean_std(sf)
                style_loss += (mse_loss(gf_mean, sf_mean) + mse_loss(gf_std, sf_std)) * args.style_weight

            total_loss = content_loss + style_loss

            (total_loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == total_steps:
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.set_description(
                f'Loss:{total_loss.item():.6f}, '
                f'Content Loss: {content_loss.item():.6f}, '
                f'Style Loss: {style_loss.item():.6f}'
            )

            running_loss += total_loss.item()
            running_closs += content_loss.item()
            running_sloss += style_loss.item()

        scheduler.step()
        steps = i + 1
        running_loss /= steps
        running_closs /= steps
        running_sloss /= steps

        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(
                f'Epoch [{epoch+1}/{args.epochs}], Total Loss: {running_loss:.4f}, '
                f'Content Loss: {running_closs:.4f}, Style Loss: {running_sloss:.4f}'
            )

        if (epoch + 1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_epoch_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_epoch_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch.to(device), style_batch.to(device), generated_images], dim=0)
                out_name = f'output_epoch_{epoch+1}_sw{args.style_weight:g}_size{final_size}.png'
                save_image(output, save_dir / out_name, nrow=args.batch_size)

            tqdm.write(f'Saved checkpoint and {out_name} in {save_dir}')

if __name__ == "__main__":
    main()