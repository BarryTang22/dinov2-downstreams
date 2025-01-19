import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from models import dinov2_segmenter
import argparse
from PIL import Image
import numpy as np

print("use gpu: ", torch.cuda.is_available())

class ADE20KModule(pl.LightningModule):
    def __init__(self, num_classes=150, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = dinov2_segmenter(num_classes=num_classes)
        
        # Convert to SyncBatchNorm for multi-GPU training
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Freeze backbone
        for param in self.model.dinov2.parameters():
            param.requires_grad = False
        
        # Add metrics for validation
        self.val_intersection = torch.zeros(num_classes)
        self.val_union = torch.zeros(num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['annotation']
        outputs = self(images)
        
        # Squeeze masks to [B, H, W] and convert to long
        masks = masks.squeeze(1).long()  # Remove channel dimension
        loss = self.criterion(outputs, masks)
        
        pred_masks = outputs.argmax(1)
        iou = self.calculate_iou(pred_masks, masks)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_iou', iou, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['annotation']
        outputs = self(images)
        masks = masks.squeeze(1).long()
        loss = self.criterion(outputs, masks)
        
        pred_masks = outputs.argmax(1)
        
        # Accumulate statistics
        for cls in range(self.hparams.num_classes):
            pred_cls = (pred_masks == cls)
            target_cls = (masks == cls)
            
            intersection = (pred_cls & target_cls).sum()
            union = (pred_cls | target_cls).sum()
            
            self.val_intersection[cls] += intersection
            self.val_union[cls] += union
        
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_start(self):
        # Reset metrics at start of validation
        self.val_intersection = torch.zeros(self.hparams.num_classes).to(self.device)
        self.val_union = torch.zeros(self.hparams.num_classes).to(self.device)
    
    def on_validation_epoch_end(self):
        # Compute IoU for each class
        valid_classes = self.val_union > 0
        ious = self.val_intersection[valid_classes].float() / self.val_union[valid_classes].float()
        mean_iou = ious.mean()
        
        self.log('val_mean_iou', mean_iou, on_epoch=True, sync_dist=True)
    
    def calculate_iou(self, pred, target, num_classes=150):
        # pred and target shape: [B, H, W]
        ious = []
        
        # Calculate IoU for each class
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum()
            union = (pred_cls | target_cls).sum()
            
            if union == 0:
                # If this class is not present in GT and prediction
                continue
            
            iou = (intersection.float() + 1e-6) / (union.float() + 1e-6)
            ious.append(iou)
        
        # Return mean IoU over all classes that appear
        return torch.stack(ious).mean() if ious else torch.tensor(0.0).to(pred.device)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        checkpoint['head_state_dict'] = self.model.head.state_dict()
        checkpoint['hyper_parameters'] = self.hparams
    
    def on_load_checkpoint(self, checkpoint):
        self.model.head.load_state_dict(checkpoint['head_state_dict'])

class ADE20KDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.CenterCrop(224),
            lambda x: torch.as_tensor(np.array(x), dtype=torch.long),
            lambda x: torch.clamp(x, 0, 149)
        ])
    
    def setup(self, stage=None):
        def transform_dataset(examples):
            examples['image'] = [self.transform(image.convert('RGB')) 
                               for image in examples['image']]
            examples['annotation'] = [self.mask_transform(mask) 
                                    for mask in examples['annotation']]
            return examples
        
        self.dataset = load_dataset("scene_parse_150")
        self.dataset = self.dataset.with_transform(transform_dataset)
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset['validation'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.wandb:
        logger = WandbLogger(project='dinov2-ade20k', name='segmentation')
    else:
        logger = pl.loggers.CSVLogger(save_dir=args.output_dir, name='logs')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='dinov2-seg-{epoch:02d}-{val_mean_iou:.2f}',
        monitor='val_mean_iou',
        mode='max',
        save_top_k=3,
        save_weights_only=True
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=2,
        strategy='ddp',
        callbacks=[checkpoint_callback],
        logger=logger,
        num_nodes=1
    )
    
    model = ADE20KModule(lr=args.lr, weight_decay=args.weight_decay)
    datamodule = ADE20KDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    
    if args.test:
        if args.checkpoint is None:
            raise ValueError("Please provide checkpoint path for testing")
        checkpoint = torch.load(args.checkpoint)
        model.model.head.load_state_dict(checkpoint['head_state_dict'])
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
