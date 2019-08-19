import numpy as np

import torch
import torchvision
import torch.autograd as autograd
import torchvision.transforms as T

import matplotlib.pyplot as plt


def preprocess(img):
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN.tolist(),
                    std=IMAGENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    
    return transform(img)


def deprocess(img, should_rescale=True):
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / IMAGENET_STD).tolist()),
        T.Normalize(mean=(-IMAGENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    
    return x_rescaled


def gram_matrix(features):
    """
    Compute the normalized Gram matrix from feature maps.
    
    Input: PyTorch Tensor of shape (N, C, H, W) representing feature maps for
           a batch of N images.
    Output: PyTorch Tensor of shape (N, C, C) representing the normalized 
            Gram matrices for the N images.
    """    
    N = features.shape[0]
    C = features.shape[1]
    H = features.shape[2]
    W = features.shape[3]
    
    features = features.reshape(N, C, H*W)
    gram_matrix = torch.matmul(features, features.transpose(1,2))

    gram_matrix = gram_matrix / (2*C*H*W)
    
    return gram_matrix


def loss(feats, layers, gm_targets, layer_weights):
    """
    Computes the MSE loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image.
    - layers: List of layer indices into feats giving the layers to include in the 
      style loss.
    - gm_targets: List of the same length as layers, where style_targets[i] is
      a PyTorch Variable giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - layer_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    mse_loss = torch.nn.MSELoss(reduction="sum")
    
    loss = 0
    for i in range(len(layers)):
        loss += layer_weights[i] * mse_loss(gm_targets[i],
                                                  gram_matrix(feats[layers[i]]))
    
    return loss


def reg_loss(generated_texture, reg_weight):
    """
    Computes the regularization loss.
    
    Inputs:
    - generated_texture: PyTorch Variable of shape (1, 3, H, W) holding an 
                         input image.
    - reg_weight: Scalar giving the weight w_t to use for the regularization
                  loss.
    Output:
    - loss: Scalar total regularization loss
    """
    # Texture has dimension 1 X 3 X H X W
    H = generated_texture.shape[2]
    W = generated_texture.shape[3]
    
    loss = torch.sum((generated_texture[:,:,range(-1, H-1),:] - generated_texture)[:,:,1:,:]**2) + torch.sum(
        (generated_texture[:,:,:,range(-1,W-1)] - generated_texture)[:,:,:,1:]**2)
    
    return reg_weight*loss


def generate_texture(model, src_texture, cnn_layers, layer_weights, 
                    reg_weight=0.02, max_iter=200, learn_rate=3, verbose=True):
    """
    Inputs:
    - model: Pretrained CNN to be used
    - src_texture: Torch tensor of source texture image
    - cnn_layers: list of indices indicating which layers to use for style loss
    - layer_weights: list of weights to use for each layer in style_layers
    - reg_weight: Regularization strength
    - max_iter: Number of iterations of gradient updates
    - learn_rate: Fixed learning rate to be used in the optimization
    
    Returns:
    - generated_texture: Generated texture image
    """
    # Find feature maps of style_img at layers specified in "style_layers"
    all_feats = []
    layer_input = src_texture
    for i, layer in enumerate(model._modules.values()):
        layer_output = layer(layer_input)
        all_feats.append(layer_output)
        layer_input = layer_output
    # Find Gram-matrices for feature maps specified in style_layers
    gm_targets = []
    for idx in cnn_layers:
        gm_targets.append(gram_matrix(all_feats[idx]))


    # Find the generated texture with gradient descent
    generated_texture = torch.rand(src_texture.shape)
    generated_texture.requires_grad = True
    
    loss_history = {'ite':[], 'loss':[]}

    optimizer = torch.optim.Adam([generated_texture], lr=learn_rate)

    for it in range(max_iter):
        # Finding the feature maps of output_img at all layers
        layer_input = generated_texture
        all_feats = []
        for i, layer in enumerate(model._modules.values()):
            layer_output = layer(layer_input)
            all_feats.append(layer_output)
            layer_input = layer_output

        # Compute loss
        loss_ = loss(all_feats, cnn_layers, gm_targets, 
                          layer_weights) + reg_loss(generated_texture, reg_weight)
        
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        
        generated_texture.data.clamp_(-1.5, 1.5)
        
        if it%10 == 0:
            loss_history['ite'].append(it)
            loss_history['loss'].append(loss_)
            if verbose:
                print("Loss is {} at iteration {}".format(loss_,it))
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    # Plot loss history
    ax1.plot(loss_history['ite'], loss_history['loss'])
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    # Show generated texture image
    ax2.imshow(deprocess(generated_texture))
    ax2.axis("off")
    plt.show()
    
    return generated_texture
