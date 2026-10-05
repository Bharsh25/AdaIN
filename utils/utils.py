from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # loads slightly cut-off files instead of crashing
Image.MAX_IMAGE_PIXELS = None            # removes the decompression bomb warning/limit
import os
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self,root,transform=None):
        super(ImageFolderDataset,self).__init__()
        self.root=root
        self.transform=transform
        self.files=list(os.listdir(root))

        self.files=[p for p in self.files if p.endswith(('.jpg','.png','.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f'Bad file skipped: {image_path} ({e})')
            return self.__getitem__((idx + 1) % len(self.files))
        if self.transform:
            image = self.transform(image)
        return image

def get_transform(size,crop,final_size):
    transform_list=[]
    if size>0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def calc_mean_std(feat, eps=1e-5):
    size=feat.size()
    assert (len(size)==4)
    batch_size, channel = size[:2]

    mean = feat.view(batch_size, channel, -1).mean(dim=2).view(batch_size, channel, 1, 1)
    var=feat.view(batch_size, channel, -1).var(dim=2,unbiased=False)+eps
    std = var.sqrt().view(batch_size, channel, 1, 1)
    return mean, std

def adaIN(content_feat, style_feat):
    #batch_size, channel,h,w
    size=content_feat.size()
    content_mean, content_std=calc_mean_std(content_feat)
    style_mean, style_std=calc_mean_std(style_feat)

    normalized_feat=(content_feat-content_mean.expand(size))/(content_std.expand(size))
    return normalized_feat*style_std.expand(size)+style_mean.expand(size)
    
