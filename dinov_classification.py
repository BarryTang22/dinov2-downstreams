import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from models import dinov2_classifier
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

print("use gpu: ", torch.cuda.is_available())

class ImageNetModule(pl.LightningModule):
    def __init__(self, num_classes=1000, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = dinov2_classifier(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
        for param in self.model.dinov2.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        _, predicted = outputs.max(1)
        acc = predicted.eq(labels).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        _, predicted = outputs.max(1)
        acc = predicted.eq(labels).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, on_epoch=True, sync_dist=True)
        
        return loss
    
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
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        _, predicted = outputs.max(1)
        acc = predicted.eq(labels).float().mean()
        
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        self.log('test_acc', acc, on_epoch=True, sync_dist=True)
        
        return {'pred': predicted, 'target': labels}
    
    def test_epoch_end(self, outputs):
        preds = torch.cat([x['pred'] for x in outputs])
        targets = torch.cat([x['target'] for x in outputs])
        
        gathered_preds = self.all_gather(preds)
        gathered_targets = self.all_gather(targets)
        
        if self.trainer.is_global_zero:
            preds_np = torch.cat([x.flatten() for x in gathered_preds]).cpu().numpy()
            targets_np = torch.cat([x.flatten() for x in gathered_targets]).cpu().numpy()
            
            os.makedirs('test_results', exist_ok=True)
            
            plt.ioff()
            cm = confusion_matrix(targets_np, preds_np)
            plt.figure(figsize=(20, 20))
            plt.imshow(cm, interpolation='nearest')
            plt.title('Confusion Matrix')
            plt.savefig('test_results/confusion_matrix.png')
            plt.close()
            
            accuracy = (preds_np == targets_np).mean()
            with open('test_results/metrics.txt', 'w') as f:
                f.write(f'Test Accuracy: {accuracy:.4f}\n')

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage=None):
        def transform_dataset(examples):
            examples['image'] = [self.transform(image.convert('RGB')) 
                               for image in examples['image']]
            return examples
        
        self.dataset = load_dataset("imagenet-1k")
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
    
    def test_dataloader(self):
        return DataLoader(
            self.dataset['validation'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='dinov2-imagenet')
    parser.add_argument('--wandb-name', type=str, default='training')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.wandb:
        logger = WandbLogger(project=args.wandb_project, name=args.wandb_name)
    else:
        logger = pl.loggers.CSVLogger(save_dir=args.output_dir, name='logs')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='dinov2-head-{epoch:02d}-{val_acc:.2f}',
        monitor='val_acc',
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
    
    model = ImageNetModule(lr=args.lr, weight_decay=args.weight_decay)
    datamodule = ImageNetDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    
    if args.test:
        if args.checkpoint is None:
            raise ValueError("Please provide checkpoint path for testing")
        checkpoint = torch.load(args.checkpoint)
        model = ImageNetModule(num_classes=1000)
        model.model.head.load_state_dict(checkpoint['head_state_dict'])
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
