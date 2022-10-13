import argparse
from torchvision.models.resnet import resnet50, resnet101, ResNet50_Weights
from torch.optim import Adam
from datasets import build_datasets
from simclr import SimCLR

def parse_option():
    parser = argparse.ArgumentParser('arguments for model training')

    parser.add_argument('--model', type=str, default='simclr',
                        help='model to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size to train')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='epochs to train')
    parser.add_argument('--num_views', type=int, default=2,
                        help='epochs to train')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='use pretrained weights')

    opt = parser.parse_args()

    return opt
        

def main():
    opt = parse_option()

    if opt.model == 'simclr':
        model = SimCLR(
                    model_backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if opt.pretrained else None), 
                    batch_size=opt.batch_size,
                    num_epochs=opt.num_epochs,
                    num_views=opt.num_views,
                    device='cuda',
                )
        
        train_loader, val_loader, test_loader = build_datasets()
        model.train(train_loader, val_loader)
            


if __name__ == '__main__':
    main()