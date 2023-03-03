import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import ObjectSegmentationDataset
import segmentation_models_pytorch as smp


def iou(prediction, target):
    _, pred = torch.max(prediction, dim=1)

    # Take number of batches and classes to iterate thru
    num_batches = prediction.shape[0]
    num_classes = prediction.shape[1]

    # Take mean IoU per class and then do the same for the batch
    ious_per_batch = []
    for batch_id in range(num_batches):
        ious_per_class = []
        # We start from index 1 since 0 is the background
        for class_id in range(1, num_classes):
            # Identify which pixels were predicted for the class and the real ones
            predicted_mask = (pred[batch_id] == class_id).int()
            actual_target = (target[batch_id] == class_id).int()
            if actual_target.sum() == 0: 
                continue

            # Calculate IoU
            intersection = (predicted_mask * actual_target).sum()
            union = (predicted_mask + actual_target).sum() - intersection
            # Append to list 
            ious_per_class.append(float(intersection) / float(union))
        mean_ious_per_batch = np.mean(ious_per_class)
        ious_per_batch.append(mean_ious_per_batch)
    return ious_per_batch


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    train_loss, train_iou = 0, 0
    for _, train_batch in enumerate(train_loader): 
        # Get output from model
        inp = train_batch['input']
        target = train_batch['target']
        output = model(inp)
        
        loss = criterion(output, target)
        mIoU = np.mean(iou(output, target))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Summing losses and mIoU
        train_loss += loss.item()
        train_iou += mIoU
    
    # Compute the avg loss for the program in the end
    train_loss /= train_loader.__len__()
    train_iou /= train_loader.__len__()
    
    return train_loss, train_iou

# Practically the same as train without gradient updates
def validate(model, device, val_loader, criterion):
    model.eval()

    validation_loss = 0 
    validation_iou = 0
    # Don't update gradients
    with torch.no_grad():
        for _, validation_batch in enumerate(val_loader): 
            inp = validation_batch['input']
            target = validation_batch['target']
            output = model(inp)

            # Get mIoU and loss 
            loss = criterion(output, target)
            mIoU = np.mean(iou(output, target))

            validation_loss += loss.item()
            validation_iou += mIoU

    validation_loss /= val_loader.__len__()
    validation_iou /= val_loader.__len__()
    return validation_loss, validation_iou


if __name__ == '__main__':

    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Directory Paths
    seg_dir = "/Users/angelmancera/Columbia/Classes/Spring_2023/ACV/Individual Project/Current_Idea/VOCdevkit/VOC2012/ImageSets/Segmentation"
    images_dir = "/Users/angelmancera/Columbia/Classes/Spring_2023/ACV/Individual Project/Current_Idea/VOCdevkit/VOC2012/JPEGImages"
    masks_dir = "/Users/angelmancera/Columbia/Classes/Spring_2023/ACV/Individual Project/Current_Idea/VOCdevkit/VOC2012/SegmentationClass"
    test_images_dir = "/Users/angelmancera/Columbia/Classes/Spring_2023/ACV/Individual Project/Current_Idea/VOCdevkit/VOC2012/TestJPEGImages"

    # Datasets
    train_dataset = ObjectSegmentationDataset(seg_dir, images_dir, masks_dir, "train")
    val_dataset = ObjectSegmentationDataset(seg_dir, images_dir, masks_dir, "val")
    test_dataset = ObjectSegmentationDataset(seg_dir, test_images_dir, "", "test")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size = 4, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size = 4, num_workers=2)

    model = smp.Unet(encoder_name="resnet34", 
                     encoder_weights="imagenet", 
                     in_channels=3,  
                     classes=21, 
                    )
    for _, param in model.encoder.named_parameters():
        param.requires_grad = False
    for _, param in model.decoder.named_parameters():
        param.requires_grad = False

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    # Train and validate the model
    train_losses = []
    train_mious = []
    validation_losses = []
    validation_mious = []
    epochs = 15
    for epoch in range(epochs+1):
        print('Epoch ' + str(epoch) + ' out of ' + str(epochs))

        train_loss, train_miou = train(model, device, train_loader, criterion, optimizer)
        validation_loss, val_miou = validate(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        train_mious.append(train_miou)
        validation_losses.append(validation_loss)
        validation_mious.append(val_miou)

        print('Train loss & Train mIoU: {:.2f} & {:.2f}'.format(train_loss, train_miou))
        print('Validation loss & Validation mIoU: {:.2f} & {:.2f}'.format(validation_loss, val_miou))
        print('---------------------------------')

    print("Finished training!")
