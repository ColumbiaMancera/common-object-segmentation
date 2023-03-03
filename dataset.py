import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

class ObjectSegmentationDataset(Dataset): 

    def __init__(self, segmentation_dir, images_dir, masks_dir, ds_type): 
        # Setting provided instance variables
        self.segmentation_dir = segmentation_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ds_type = ds_type

        # Set image paths instance vars
        self.set_image_paths()

        # Set dataset length
        self.ds_length = len(self.img_paths)

        # Also initialize image transforms here
        # First get mean and std of all rgb images
        mean_rgb = [0.485, 0.456, 0.406]
        std_rgb = [0.229, 0.224, 0.225]
        self.image_transform_rgb = transforms.Compose([transforms.ToTensor(), 
                                                       transforms.Normalize(mean_rgb, std_rgb)])

    def set_image_paths(self): 
        # Gets image paths or mask paths depending on boolean value. 
        file_codes = self.read_file_into_list()
        # Add prefix to file_codes for both directories. 
        self.img_paths = [self.images_dir + "/" + file_code + ".jpg" for file_code in file_codes]
        
        if self.ds_type != "test": 
            self.mask_paths = [self.masks_dir + "/" + file_code + ".png" for file_code in file_codes]

    
    def generate_segmentation_mask(self, mask): 
        segmentation_mask = np.zeros(mask.shape, dtype=np.int64)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[np.all(mask == label, axis=-1)] = label_index 
        return segmentation_mask # .transpose(2, 1, 0)

    def read_file_into_list(self): 
        file_codes = []
        seg_path = self.segmentation_dir + "/" + self.ds_type + ".txt"
        with open(seg_path) as file_listing: 
            for line in file_listing: 
                file_codes.append(line.strip())
        return file_codes
        
    def __len__(self): 
        return self.ds_length
    
    def __getitem__(self, idx):
        # Get RGB image and apply transform
        rgb_path = self.img_paths[idx]
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.resize(rgb_image, dsize=(576, 576), interpolation=cv2.INTER_NEAREST)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = self.image_transform_rgb(rgb_image)

        if self.ds_type != "test": 
            # Get image mask
            gt_path = self.mask_paths[idx]
            gt_mask = cv2.imread(gt_path)
            gt_mask = cv2.resize(gt_mask, dsize=(576, 576), interpolation=cv2.INTER_NEAREST)
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
            # Converts segmentation mask to proper format
            gt_mask = self.generate_segmentation_mask(gt_mask)[:,:, -1]
            gt_mask = torch.LongTensor(gt_mask)
            # gt_mask = torch.movedim(gt_mask, 2, 0)
            return {'input': rgb_image, 'target': gt_mask}
        else: 
            return {'input': rgb_image}
    
if __name__ == "__main__": 
    seg_dir = "/Users/angelmancera/Columbia/Classes/Spring_2023/ACV/Individual Project/Current_Idea/VOCdevkit/VOC2012/ImageSets/Segmentation"
    images_dir = "/Users/angelmancera/Columbia/Classes/Spring_2023/ACV/Individual Project/Current_Idea/VOCdevkit/VOC2012/JPEGImages"
    masks_dir = "/Users/angelmancera/Columbia/Classes/Spring_2023/ACV/Individual Project/Current_Idea/VOCdevkit/VOC2012/SegmentationClass"
    ds_type = "train"
    seg_ds = ObjectSegmentationDataset(seg_dir, images_dir, masks_dir, ds_type)

    # To run a small test of our seg_ds
    print("seg_ds size:", len(seg_ds))
    sample = seg_ds[np.random.randint(len(seg_ds))]
    rgb_image = sample['input'].numpy()
    print("input shape:", rgb_image.shape)
    if seg_ds.ds_type != "test":
        gt_mask = sample['target'].numpy()
        print("target shape:", gt_mask.shape)
