"""

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms, utils
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

def arg_parser():
    import argparse
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str.Default is vgg16')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be saved on current Directory.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float,default is 0.01')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier as int')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
     
    return args



    

def data_loader():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])



    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)
    class_to_idx=train_data.class_to_idx
    return trainloader,vloader,testloader,class_to_idx

def select_model(arch="vgg16"):
    if (arch == "vgg16"):
        model = models.vgg16(pretrained=True)
    elif(arch == "alexnet"):
        model=models.alexnet(pretrained=True)
    elif(arch=="densenet121"):
        model=models.densenet121(pretrained=True)
    else:
        model= None
    #Freeze parameters so we don't backprop through them
    for param in model.parameters(): 
        param.requires_grad = False
    return model

def validation(model, valid_loader, criterion):
    model.to ('cuda')
    
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        #equality = (labels.data == ps.max(dim=1)[1])
        top_p, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def train(model, trainloader, validloader,device, criterion, optimizer, epochs,print_every):
    model.to (device)
    #epochs = 5
    print_every = 20
    steps = 0
    train_loss_list=[]
    valid_loss_list=[]

    for e in range (epochs): 
        running_loss = 0
        for ii, (inputs, labels) in enumerate (trainloader):
            steps += 1
        
            inputs = inputs.to(device)
            labels= labels.to(device)
            optimizer.zero_grad () 
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
        
            # Forward and backward passes
            outputs = model.forward (inputs) #calculating output
            loss = criterion (outputs, labels) #calculating loss
            loss.backward () 
            optimizer.step () #performs single optimization step 
        
            running_loss += loss.item () # loss.item () returns scalar value of Loss function
        
            if steps % print_every == 0:
                model.eval () #switching to evaluation mode so that dropout is turned off
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
                
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                    "Valid Accuracy: {:.3f}%".format(accuracy/len(validloader)*100))
                train_loss_list.append(running_loss)
                valid_loss_list.append(valid_loss)
                running_loss = 0
                
                # Make sure training is back on
                model.train()


def test(model, testloader, criterion):
    model.eval () #switching to evaluation mode so that dropout is turned off
            
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        valid_loss, accuracy = validation(model, testloader, criterion)
            
            
    print("Valid Loss: {:.3f}.. ".format(valid_loss/len(testloader)),
        "Valid Accuracy: {:.3f}%".format(accuracy/len(testloader)*100))   
                            
def save_model(model,class_to_idx,file_path=None):
    checkpoint = {
              'pretrain_model':  type(model),
              'state_dict': model.state_dict(),
              'classifier':model.classifier,
              'class_to_idx':class_to_idx}
    file_path=file_path+'/checkpoint.pth'
    torch.save(checkpoint, 'checkpoint.pth')

if __name__ == "__main__":
    arg=arg_parser()
    if arg.hidden_units  == None :
        print ("0")
    print(arg)
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    trainloader,vloader,testloader,class_to_idx=data_loader()
    if arg.arch:
        model=select_model(arg.arch)
    else:
        model = select_model()
    if model: 
        if arg.hidden_units != None : 
            hidden_units= arg.hidden_units
        else:
            hidden_units= 4096
        
        if (arg.arch == "vgg16"):
            in_feature = model.classifier[0].in_features
        elif(arg.arch == "alexnet"):
            in_feature=9216
        elif(arg.arch=="densenet121"):
            in_feature=model.classifier.in_features
        
        
        
        classifier = nn.Sequential(OrderedDict ([
                                ('fc1', nn.Linear (in_feature, hidden_units)),
                                ('relu1', nn.ReLU ()),
                                ('dropout1', nn.Dropout (p = 0.5)),
                                ('fc2', nn.Linear (hidden_units, 102)),
                                ('output', nn.LogSoftmax (dim =1))
                                ]))
        model.classifier = classifier
        criterion = nn.NLLLoss ()

        if arg.learning_rate  != None: 
            lr= arg.learning_rate
        else:
            lr=0.001
        optimizer = optim.Adam (model.classifier.parameters (), lr =lr)
        print (arg.gpu, arg.gpu == None)
        if arg.gpu :
            model.to('cuda')
            device='cuda'
        else:
            model.to('cpu')    
            device='cpu'
        print(device)
        if arg.epochs != None:
            epoch=arg.epochs
        else:
            epoch=5        
        train(model,trainloader,vloader,device,criterion,optimizer,epoch,20)
        test(model, testloader, criterion)
        if arg.save_dir != None :
            file_path=arg.save_dir
        else:
            file_path='.'
        save_model(model,class_to_idx,file_path)      
    else :
        print("currently only 3 models are supported vgg16,densenet121,alexnet")
    #print(arg)
    