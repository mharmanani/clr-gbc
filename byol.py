"""
@author: Sana Arastehfar
"""
import torch
import time
import os

import numpy as np
import torch.optim as optim

from datetime import datetime
from byol_pytorch import BYOL
from torchvision.models.resnet import resnet101, resnet18
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.functional import precision_recall, f1_score, auroc
from datasets import HistologyDataset


def train_byol_model(model_backbone, epochs=10, batch_size=8, validation_split=0.2, random_seed=42, shuffle_dataset=True):
    # preparing the dataset
    train_path = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/Train'
    train_annot = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/train_annotations_binary.csv'
    
    test_path = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/Test'
    test_annot = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/test_annotations_binary.csv'
    
    # check whether the dataset exists
    data_dir_exists = os.path.exists(train_path)
    assert data_dir_exists, '[!] The directory "data/train" containing the training files does not exits!'
    
    annotations_exists = os.path.exists(train_annot)
    assert annotations_exists, '[!] The annotation file "data/train_annot.csv" does not exist!'
    
    data_dir_contents = len(os.listdir(train_path)) > 0
    assert data_dir_contents, '[!] The dataset folder is empty!'
    
    
    # creating the checkpoints directory if it does not exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('checkpoints/byol', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/byol', exist_ok=True)

    # Creating data indices for training and validation splits:
    # dataset_size = len(os.listdir(train_path))
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]

    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)

    # Create the train data loader
    train_dataset = HistologyDataset(annotations_file=train_annot, 
                                     img_dir=train_path, 
                                     transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the validation data loader
    val_dataset = HistologyDataset(annotations_file=test_annot, 
                                   img_dir=test_path, 
                                   transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]))
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f' Running on {device}')

    # defining the model
    model = BYOL(
        model_backbone,
        image_size= 224,
        hidden_layer='avgpool'
    ).to(device)

    # defnining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    run_name = f'BYOL-{datetime.now().year}-{datetime.now().month}-{datetime.now().day}--{datetime.now().hour}-{datetime.now().minute}'
    
    best_val_loss = float('inf')
    
    training_losses = []
    validation_losses = []
    
    # training the model
    for epoch in range(epochs):
        start = time.time()
        avg_train_loss = 0.
        for image, _ in train_loader:
            image = image.to(device)

            optimizer.zero_grad()
            
            loss = model(image)
            
            loss.backward()
            optimizer.step()
            
            model.update_moving_average()
            
            avg_train_loss += loss.item()
            
        avg_train_loss = avg_train_loss / len(train_loader)
        avg_val_loss = 0

        # validating the model
        with torch.inference_mode():
            for sample in val_loader:
                image = sample[0].to(device)
                
                loss = model(image)
                
                avg_val_loss += loss.item()
                
            avg_val_loss = avg_val_loss / len(val_loader)
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'checkpoints/byol/{run_name}.ckpt')
            msg = ' - [CHECKPOINT]'
        else:
            msg = ''
            
        end = time.time()
        
        training_losses.append(avg_train_loss)
        validation_losses.append(avg_val_loss)

        print(f'[*] Epoch: {epoch} - Avg Train Loss: {avg_train_loss:.3f} - '
              f'Avg Val Loss: {avg_val_loss:.3f} - Elapsed: {end - start:.2f}' + msg)
    
    # saving the learning curve of the BYOL model
    np.savetxt(f'logs/byol/{run_name}_train_losses.txt', np.array(training_losses))
    np.savetxt(f'logs/byol/{run_name}_val_losses.txt', np.array(validation_losses))

    return model

def byol_based_classifier(model_backbone, byol_weights, epochs=10, batch_size=8, validation_split=0.2, random_seed=42, shuffle_dataset=True, use_embeddings=True):
    # preparing the dataset
    train_path = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/Train'
    train_annot = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/annotations_binary.csv'
    
    test_path = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/Test'
    test_annot = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/test_annotations_binary.csv'
    
    # check whether the dataset exists
    data_dir_exists = os.path.exists(train_path)
    assert data_dir_exists, '[!] The directory "data/train" containing the training files does not exits!'
    
    annotations_exists = os.path.exists(train_annot)
    assert annotations_exists, '[!] The annotation file "data/train_annot.csv" does not exist!'
    
    data_dir_contents = len(os.listdir(train_path)) > 0
    assert data_dir_contents, '[!] The dataset folder is empty!'
    
    
    # creating the checkpoints directory if it does not exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('checkpoints/byol', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/byol', exist_ok=True)
    
    # Creating data indices for training and validation splits:
    # dataset_size = len(os.listdir(train_path))
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]

    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)

    # Create the train data loader
    train_dataset = HistologyDataset(annotations_file=train_annot, 
                                     img_dir=train_path, 
                                     transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the validation data loader
    val_dataset = HistologyDataset(annotations_file=test_annot, 
                                   img_dir=test_path, 
                                   transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]))
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f' Running on {device}')

    # defining the model
    model = BYOL(
        model_backbone,
        image_size= 224,
        hidden_layer='avgpool'
    ).to(device)
    
    model.load_state_dict(torch.load(byol_weights))
    # model.eval()
    
    # defining the classifier head
    first_layer_size = 512 if use_embeddings else 256
    classifier = torch.nn.Sequential(
        torch.nn.Linear(first_layer_size, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 2)
    ).to(device)
    
    # defnining the optimizer
    optimizer = optim.Adam(list(classifier.parameters()) + list(model.parameters()), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    run_name = f'BYOL-Classifier-{datetime.now().year}-{datetime.now().month}-{datetime.now().day}--{datetime.now().hour}-{datetime.now().minute}'
    
    best_val_loss = float('inf')
    
    training_losses = []
    training_accs = []
    training_precisions = []
    training_recalls = []
    training_f1_scores = []
    training_aucroc = []
    
    validation_losses = []
    validation_accs = []
    validation_precisions = []
    validation_recalls = []
    validation_f1_scores = []
    validation_aucroc = []
    
    # training the model
    for epoch in range(epochs):
        start = time.time()
        avg_train_loss = 0.
        avg_train_acc = 0.
        avg_train_prec = 0.
        avg_train_rec = 0.
        avg_train_f1 = 0.
        avg_train_aucroc = 0
        model.train()
        for image, label in train_loader:
            image, label = image.to(device), label.type(torch.int64).to(device)
            
            optimizer.zero_grad()
            
            byol_out = model(image, return_embedding=True)
            if use_embeddings:
               byol_out = byol_out[1]
            else:
               byol_out = byol_out[0]
   
            out = classifier(byol_out)
            
            loss = loss_fn(out, label)
            
            predictions = out.argmax(dim=-1).detach().cpu()
            accuracy = (predictions == label.cpu()).sum() / len(label)
            precision, recall = precision_recall(predictions, label.cpu())
            f1 = f1_score(predictions, label.cpu())
            auc_roc = auroc(out.detach().cpu(), label.cpu(), num_classes=2)
            
            loss.backward()
            optimizer.step()
            
            model.update_moving_average()
            
            avg_train_loss += loss.item()
            avg_train_acc += accuracy
            avg_train_prec += precision.item()
            avg_train_rec += recall.item()
            avg_train_f1 += f1.item()
            avg_train_aucroc += auc_roc.item()

        avg_train_loss = avg_train_loss / len(train_loader)
        avg_train_acc = avg_train_acc / len(train_loader)
        avg_train_prec = avg_train_prec / len(train_loader)
        avg_train_rec = avg_train_rec / len(train_loader)
        avg_train_f1 = avg_train_f1 / len(train_loader)
        avg_train_aucroc = avg_train_aucroc / len(train_loader)
        
        avg_val_loss = 0.
        avg_val_acc = 0.
        avg_val_prec = 0.
        avg_val_rec = 0.
        avg_val_f1 = 0.
        avg_val_aucroc = 0.
        # validating the model
        model.eval()
        with torch.inference_mode():
            for image, label in val_loader:
                image, label = image.to(device), label.type(torch.int64).to(device)
                
                byol_out = model(image, return_embedding=True)
                if use_embeddings:
                    byol_out = byol_out[1]
                else:
                    byol_out = byol_out[0]

                out = classifier(byol_out)
                
                loss = loss_fn(out, label)
                
                predictions = out.argmax(dim=-1).cpu()
                accuracy = (predictions == label.cpu()).sum() / len(label)
                precision, recall = precision_recall(predictions, label.cpu())
                f1 = f1_score(predictions, label.cpu())
                auc_roc = auroc(out.cpu(), label.cpu(), num_classes=2)
                
                avg_val_loss += loss.item()
                avg_val_acc += accuracy
                avg_val_prec += precision.item()
                avg_val_rec += recall.item()
                avg_val_f1 += f1.item()
                avg_val_aucroc += auc_roc.item()
                
            avg_val_loss = avg_val_loss / len(val_loader)
            avg_val_acc = avg_val_acc / len(val_loader)
            avg_val_prec = avg_val_prec / len(val_loader)
            avg_val_rec = avg_val_rec / len(val_loader)
            avg_val_f1 = avg_val_f1 / len(val_loader)
            avg_val_aucroc = avg_val_aucroc / len(val_loader)
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(classifier.state_dict(), f'checkpoints/byol/{run_name}.ckpt')
            msg = ' - [CHECKPOINT]'
        else:
            msg = ''
            
        end = time.time()
        
        training_losses.append(avg_train_loss)
        training_accs.append(avg_train_acc)
        training_precisions.append(avg_train_prec)
        training_recalls.append(avg_train_rec)
        training_f1_scores.append(avg_train_f1)
        training_aucroc.append(avg_train_aucroc)
        
        validation_losses.append(avg_val_loss)
        validation_accs.append(avg_val_acc)
        validation_precisions.append(avg_val_prec)
        validation_recalls.append(avg_val_rec)
        validation_f1_scores.append(avg_val_f1)
        validation_aucroc.append(avg_val_aucroc)

        print(f'[*] Epoch: {epoch} \n'
              f'Train Loss: {avg_train_loss:.3f} - Train Acc: {avg_train_acc:.3f} - Train Precision: {avg_train_prec:.2f} - Train Recall: {avg_train_rec:.2f} - Train F1: {avg_train_f1:.2f} - Train AUROC: {avg_train_aucroc:.2f}\n'
              f'Val Loss: {avg_val_loss:.3f} - Val Acc: {avg_val_acc:.3f} - Val Precision: {avg_val_prec:.2f} - Val Recall: {avg_val_rec:.2f} - Val F1: {avg_val_f1:.2f} - Val AUROC: {avg_val_aucroc:.2f}\n'
              f'Elapsed: {end - start:.2f}' + msg)
    
    # saving the learning curve of the BYOL model
    np.savetxt(f'logs/byol/{run_name}_train_losses.txt', np.array(training_losses))
    np.savetxt(f'logs/byol/{run_name}_train_accs.txt', np.array(training_accs))
    np.savetxt(f'logs/byol/{run_name}_train_precisions.txt', np.array(training_precisions))
    np.savetxt(f'logs/byol/{run_name}_train_recalls.txt', np.array(training_recalls))
    np.savetxt(f'logs/byol/{run_name}_train_f1_scores.txt', np.array(training_f1_scores))
    np.savetxt(f'logs/byol/{run_name}_train_aucroc.txt', np.array(training_aucroc))
    
    np.savetxt(f'logs/byol/{run_name}_val_losses.txt', np.array(validation_losses))
    np.savetxt(f'logs/byol/{run_name}_val_accs.txt', np.array(validation_accs))
    np.savetxt(f'logs/byol/{run_name}_val_precisions.txt', np.array(validation_precisions))
    np.savetxt(f'logs/byol/{run_name}_val_recalls.txt', np.array(validation_recalls))
    np.savetxt(f'logs/byol/{run_name}_val_f1_scores.txt', np.array(validation_f1_scores))
    np.savetxt(f'logs/byol/{run_name}_val_aucroc.txt', np.array(validation_aucroc))

    return model

def resnet_classifier(model_backbone, epochs=10, batch_size=8, validation_split=0.2, random_seed=42, shuffle_dataset=True):
    # preparing the dataset
    train_path = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/Train'
    train_annot = 'data/NCT-CRC-HE-100K-NONORM-MODIFIED/annotations_binary.csv'
    
    # check whether the dataset exists
    data_dir_exists = os.path.exists(train_path)
    assert data_dir_exists, '[!] The directory "data/train" containing the training files does not exits!'
    
    annotations_exists = os.path.exists(train_annot)
    assert annotations_exists, '[!] The annotation file "data/train_annot.csv" does not exist!'
    
    data_dir_contents = len(os.listdir(train_path)) > 0
    assert data_dir_contents, '[!] The dataset folder is empty!'
    
    
    # creating the checkpoints directory if it does not exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('checkpoints/resnet', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/resnet', exist_ok=True)
    
    # Creating data indices for training and validation splits:
    dataset_size = len(os.listdir(train_path))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create the train data loader
    train_dataset = HistologyDataset(annotations_file=train_annot, 
                                     img_dir=train_path, 
                                     transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the validation data loader
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

    # use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f' Running on {device}')
    
    # defining the model
    model = model_backbone.to(device)
    
    model.train()
    
    # defining the classifier head
    classifier = torch.nn.Sequential(
        torch.nn.Linear(1000, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 2)
    ).to(device)
    
    # defnining the optimizer
    optimizer = optim.Adam(list(classifier.parameters()) + list(model.parameters()), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    run_name = f'ResNet-Classifier-{datetime.now().year}-{datetime.now().month}-{datetime.now().day}--{datetime.now().hour}-{datetime.now().minute}'
    
    best_val_loss = float('inf')
    
    training_losses = []
    training_accs = []
    training_precisions = []
    training_recalls = []
    training_f1_scores = []
    training_aucroc = []
    
    validation_losses = []
    validation_accs = []
    validation_precisions = []
    validation_recalls = []
    validation_f1_scores = []
    validation_aucroc = []
    
    # training the model
    for epoch in range(epochs):
        start = time.time()
        avg_train_loss = 0.
        avg_train_acc = 0.
        avg_train_prec = 0.
        avg_train_rec = 0.
        avg_train_f1 = 0.
        avg_train_aucroc = 0

        for image, label in train_loader:
            image, label = image.to(device), label.type(torch.int64).to(device)

            optimizer.zero_grad()
            
            resnet_out = model(image)
            
            out = classifier(resnet_out)

            loss = loss_fn(out, label)
            
            predictions = out.argmax(dim=-1).detach().cpu()
            accuracy = (predictions == label.cpu()).sum() / len(label)
            precision, recall = precision_recall(predictions, label.cpu())
            f1 = f1_score(predictions, label.cpu())
            auc_roc = auroc(out.detach().cpu(), label.cpu(), num_classes=2)
            
            loss.backward()
            optimizer.step()
            
            avg_train_loss += loss.item()
            avg_train_acc += accuracy.item()
            avg_train_prec += precision.item()
            avg_train_rec += recall.item()
            avg_train_f1 += f1.item()
            avg_train_aucroc += auc_roc.item()

        avg_train_loss = avg_train_loss / len(train_loader)
        avg_train_acc = avg_train_acc / len(train_loader)
        avg_train_prec = avg_train_prec / len(train_loader)
        avg_train_rec = avg_train_rec / len(train_loader)
        avg_train_f1 = avg_train_f1 / len(train_loader)
        avg_train_aucroc = avg_train_aucroc / len(train_loader)
        
        avg_val_loss = 0.
        avg_val_acc = 0.
        avg_val_prec = 0.
        avg_val_rec = 0.
        avg_val_f1 = 0.
        avg_val_aucroc = 0.

        # validating the model
        with torch.inference_mode():
            for image, label in val_loader:
                image, label = image.to(device), label.type(torch.int64).to(device)
                
                resnet_out = model(image)

                out = classifier(resnet_out)
                
                loss = loss_fn(out, label)
                
                predictions = out.argmax(dim=-1).cpu()
                accuracy = (predictions == label.cpu()).sum() / len(label)
                precision, recall = precision_recall(predictions, label.cpu())
                f1 = f1_score(predictions, label.cpu())
                auc_roc = auroc(out.detach().cpu(), label.cpu(), num_classes=2)
                
                avg_val_loss += loss.item()
                avg_val_acc += accuracy
                avg_val_prec += precision.item()
                avg_val_rec += recall.item()
                avg_val_f1 += f1.item()
                avg_val_aucroc += auc_roc.item()
                
            avg_val_loss = avg_val_loss / len(val_loader)
            avg_val_acc = avg_val_acc / len(val_loader)
            avg_val_prec = avg_val_prec / len(val_loader)
            avg_val_rec = avg_val_rec / len(val_loader)
            avg_val_f1 = avg_val_f1 / len(val_loader)
            avg_val_aucroc = avg_val_f1 / len(val_loader)
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(classifier.state_dict(), f'checkpoints/resnet/{run_name}.ckpt')
            msg = ' - [CHECKPOINT]'
        else:
            msg = ''
            
        end = time.time()
        
        training_losses.append(avg_train_loss)
        training_accs.append(avg_train_acc)
        training_precisions.append(avg_train_prec)
        training_recalls.append(avg_train_rec)
        training_f1_scores.append(avg_train_f1)
        training_aucroc.append(avg_train_aucroc)
        
        validation_losses.append(avg_val_loss)
        validation_accs.append(avg_val_acc)
        validation_precisions.append(avg_val_prec)
        validation_recalls.append(avg_val_rec)
        validation_f1_scores.append(avg_val_f1)
        validation_aucroc.append(avg_val_aucroc)

        print(f'[*] Epoch: {epoch} \n'
              f'Train Loss: {avg_train_loss:.3f} - Train Acc: {avg_train_acc:.3f} - Train Precision: {avg_train_prec:.2f} - Train Recall: {avg_train_rec:.2f} - Train F1: {avg_train_f1:.2f} - Train AUROC: {avg_train_aucroc:.2f}\n'
              f'Val Loss: {avg_val_loss:.3f} - Val Acc: {avg_val_acc:.3f} - Val Precision: {avg_val_prec:.2f} - Val Recall: {avg_val_rec:.2f} - Val F1: {avg_val_f1:.2f} - Val AUROC: {avg_val_aucroc:.2f}\n'
              f'Elapsed: {end - start:.2f}' + msg)
    
    # saving the learning curve of the BYOL model
    np.savetxt(f'logs/resnet/{run_name}_train_losses.txt', np.array(training_losses))
    np.savetxt(f'logs/resnet/{run_name}_train_accs.txt', np.array(training_accs))
    np.savetxt(f'logs/resnet/{run_name}_train_precisions.txt', np.array(training_precisions))
    np.savetxt(f'logs/resnet/{run_name}_train_recalls.txt', np.array(training_recalls))
    np.savetxt(f'logs/resnet/{run_name}_train_f1_scores.txt', np.array(training_f1_scores))
    np.savetxt(f'logs/resnet/{run_name}_train_aucroc.txt', np.array(training_aucroc))
    
    np.savetxt(f'logs/resnet/{run_name}_val_losses.txt', np.array(validation_losses))
    np.savetxt(f'logs/resnet/{run_name}_val_accs.txt', np.array(validation_accs))
    np.savetxt(f'logs/resnet/{run_name}_val_precisions.txt', np.array(validation_precisions))
    np.savetxt(f'logs/resnet/{run_name}_val_recalls.txt', np.array(validation_recalls))
    np.savetxt(f'logs/resnet/{run_name}_val_f1_scores.txt', np.array(validation_f1_scores))
    np.savetxt(f'logs/resnet/{run_name}_val_aucroc.txt', np.array(validation_aucroc))

    return model
    
    
if __name__ == '__main__':
    backbone = resnet18(pretrained=True)
    
    epochs = 50
    batch_size = 50
    
    # parameters for the classifier
    # if use_embedding is True, the 2048 dimensional embeddings are output and if it is False, the 256 dimensional projections are used
    use_embeddings = True
    byol_weights = 'checkpoints/byol/BYOL-2022-11-26--14-58.ckpt'
    
    # train_byol_model(model_backbone=backbone, epochs=epochs, batch_size=batch_size)
    byol_based_classifier(model_backbone=backbone, byol_weights=byol_weights, epochs=epochs, batch_size=batch_size, use_embeddings=use_embeddings)
    # resnet_classifier(model_backbone=backbone, epochs=epochs, batch_size=batch_size)
    
