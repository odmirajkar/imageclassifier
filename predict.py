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
    # Define a parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # Point towards image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='Point to impage file for prediction.',
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        required=True)
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping from categories to real names.')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')

    # Parse args
    args = parser.parse_args()
    
    return args

def load_checkpoint(checkpoint='checkpoint.pth'):
    """
    function to laod model and return it 

    """
    
    checkpoint = torch.load(checkpoint)

    print(checkpoint['pretrain_model'])
    if ('torchvision.models.alexnet.AlexNet' in str(checkpoint['pretrain_model']) ):

        model = models.alexnet(pretrained=True)
    elif ('torchvision.models.densenet.DenseNet' in str(checkpoint['pretrain_model']) ):
        model = models.densenet121(pretrained=True)    
    else:
        model= models.vgg16(pretrained=True)       
    model.classifier=checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])   
    model.class_to_idx=checkpoint['class_to_idx']#it should be saved in file and loaded from file 
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img_w,img_h=img.size
    #a_ratio=
    if img_w > img_h :
        img.thumbnail(size=[img_w/img_h * 256,256])
    else:
        
        img.thumbnail(size=[256,img_h/img_w * 256])
        
    center = img.size[0]/2, img.size[1]/2
    left, top, right, bottom = center[0]-(224/2), center[1]-(224/2), center[0]+(224/2), center[1]+(224/2)
    img=img.crop((left, top, right, bottom))
    
    np_image = np.array(img)/255
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    trch_img=torch.from_numpy(np_image).type(torch.float)
    return trch_img#np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    #image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk,device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    preprocess_img=process_image(image_path)
    preprocess_img.unsqueeze_(0)#to avoid expected stride to be a single integer value or a list of 1 values
    #preprocess_img=torch.from_numpy(np.expand_dims(process_image(image_path), 
    #                                              axis=0)).type(torch.FloatTensor)
    model.to(device)
    preprocess_img.to(device)
    #preprocess_img.type(torch.DoubleTensor)
    model.eval()
    
    with torch.no_grad():
        log_op=model.forward(preprocess_img)
    #print(log_op)
    op = torch.exp(log_op)
    top_probs, top_labels = op.topk(topk)
    return top_probs,top_labels

def load_cat():
    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name    


if __name__ == "__main__":
    arg = arg_parser()
    model=load_checkpoint(arg.checkpoint)
    if arg.top_k:
        top_k=arg.top_k
    else:
        top_k=5    
    if arg.gpu:
        device='cuda'
    else:
        device='cpu'        
    
    cat_to_name=load_cat()
    
    top_probs,top_labels  =predict(arg.image,model,top_k,device)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labels = top_labels.detach().numpy().tolist()[0]
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    for flower,prob in zip(top_flowers,top_probs):
        print("{} => {}%".format(flower,prob*100))