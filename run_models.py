import argparse
from byol import train_byol_model
import torch
from torchvision.models.resnet import resnet18, resnet50, \
    ResNet18_Weights, ResNet50_Weights
from torch.optim import Adam
from datasets import build_datasets

# Import models
from simclr import SimCLR
from supcon import SupCon
from byol import train_byol_model, test_byol_model, train_byol_classifier
from resnet import ResNetWrapper
from cnn import LeNetClassifier
from cnn import train as train_cnn
from cnn import test as test_cnn

def parse_option():
    parser = argparse.ArgumentParser('arguments for model training')

    parser.add_argument('--model', type=str, default='simclr',
                        help='model to train')
    parser.add_argument('--mode', type=str, default='train',
                        help='determine wether to train or test model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size to train')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='epochs to train')
    parser.add_argument('--num_views', type=int, default=2,
                        help='epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate')                
    parser.add_argument('--baseline', type=str, default='resnet18',
                        help='the model backbone to use')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='use pretrained weights')
    parser.add_argument('--from_epoch', type=int, default=0,
                        help='resume training by loading a checkpoint')

    opt = parser.parse_args()

    return opt
        

def main():
    opt = parse_option()
    print(opt)

    if opt.model == 'simclr':
        # Create data loaders for training with stochastic augmentation
        train_loader, val_loader, _ = build_datasets(batch_size=opt.batch_size, augment_views=True)
        
        # Initialize the SimCLR model instanc with appropriate hyperparameters
        simclr = SimCLR(
                    model_backbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if opt.pretrained else None), 
                    batch_size=opt.batch_size,
                    num_epochs=opt.num_epochs,
                    num_views=opt.num_views,
                    learning_rate=opt.learning_rate,
                    device='cuda',
                )
        
        if opt.from_epoch > 0: # load weights from previous training tasks
            print('loading weights from: checkpoints/simclr/{0}.pth'.format(opt.from_epoch))
            simclr.model.load_state_dict(torch.load('checkpoints/simclr/{0}.pth'.format(opt.from_epoch)))

        if opt.mode == 'train': # launch model training
            simclr.train(train_loader=train_loader, val_loader=val_loader)
        
        # Remove stochastic augmentation for downstream classification tasks
        train_loader, val_loader, test_loader = build_datasets(batch_size=opt.batch_size, augment_views=False)
        simclr.train_clf_head(train_loader=train_loader, val_loader=val_loader, num_epochs=200)
        
        # Compute the test metrics
        test_metrics = simclr.test(test_loader=test_loader, batch_size=opt.batch_size)
        print(test_metrics)

    elif opt.model == 'byol':
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if opt.pretrained else None)
        train_loader, val_loader, test_loader = build_datasets(batch_size=opt.batch_size, augment_views=False)
        if opt.mode == 'train':
            train_byol_model(
                model_backbone=backbone, 
                epochs=opt.num_epochs,
                batch_size=opt.batch_size)
        elif opt.mode == 'test':
            train_byol_classifier(backbone, train_loader, val_loader, load_weights=opt.from_epoch > 0, from_epoch=opt.from_epoch)
            test_accuracy = test_byol_model(model_backbone=backbone, load_weights=opt.from_epoch > 0, from_epoch=opt.from_epoch)
            print(test_accuracy)
        

    elif opt.model == 'baseline':
        train_loader, val_loader, test_loader = build_datasets(batch_size=opt.batch_size, augment_views=False)
        
        if opt.baseline == 'resnet18':
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if opt.pretrained else None)
            model = ResNetWrapper(resnet).to('cuda')

            if opt.from_epoch > 0:
                print('loading weights from: checkpoints/resnet/{0}.pth'.format(opt.from_epoch))
                model.clf.load_state_dict(torch.load('checkpoints/resnet/{0}.pth'.format(opt.from_epoch)))

            if opt.mode == 'train':
                model.finetune(train_loader=train_loader, val_loader=val_loader)
            
            if opt.mode == 'test':
                test_accuracy = model.test(test_loader=test_loader)
                print(test_accuracy)

        elif opt.baseline == 'cnn':
            model = LeNetClassifier()
            train_cnn(model, train_loader, val_loader)
            test_accuracy = test_cnn(model, test_loader=test_loader)
            print(test_accuracy)
        
        elif opt.baseline == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if opt.pretrained else None) 
            model = ResNetWrapper(resnet).to('cuda')
            model.finetune(train_loader=train_loader, val_loader=val_loader)
            test_accuracy = model.test(test_loader=test_loader)
            print(test_accuracy)    
    
    elif opt.model == 'supcon':
        # Create data loaders for training with stochastic augmentation
        train_loader, val_loader, _ = build_datasets(batch_size=opt.batch_size, augment_views=True)
        
        # Initialize the SimCLR model instanc with appropriate hyperparameters
        supcon = SupCon(
                    model_backbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if opt.pretrained else None), 
                    batch_size=opt.batch_size,
                    num_epochs=opt.num_epochs,
                    num_views=opt.num_views,
                    learning_rate=opt.learning_rate,
                    device='cuda',
                )
        
        if opt.from_epoch > 0: # load weights from previous training tasks
            print('loading weights from: checkpoints/supcon/{0}.pth'.format(opt.from_epoch))
            supcon.model.load_state_dict(torch.load('checkpoints/supcon/{0}.pth'.format(opt.from_epoch)))

        if opt.mode == 'train': # launch model training
            supcon.train(train_loader=train_loader, val_loader=val_loader)
        
        # Remove stochastic augmentation for downstream classification tasks
        train_loader, val_loader, test_loader = build_datasets(batch_size=opt.batch_size, augment_views=False)
        supcon.train_clf_head(train_loader=train_loader, val_loader=val_loader, num_epochs=200)
        
        # Compute the test metrics
        test_metrics = supcon.test(test_loader=test_loader, batch_size=opt.batch_size)
        print(test_metrics)
    else:
        print('None selected')
            


if __name__ == '__main__':
    main()