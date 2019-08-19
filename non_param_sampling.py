import numpy as np
from scipy import ndimage

def synthesize_texture(src_texture, output_shape=(65, 65), window_sz=(10, 10), 
                        random_init=True, verbose=False):
    """
    Synthesizes texture following the Efros-Leung algorithm

    Inputs:
    - src_texture: Sample texture image as a numpy array
    - window_sz: Size of window used to find best matching patches
    - output_shape: Size of final generated texture, should be larger than that
                    of sample texture
    - random_init: If True, use random first pixel. 
                   If False, use part of the source texture.
    - verbose: Whether to print progress
    
    Output: Generated texture image as a numpy array
    """
    src_H, src_W, C = src_texture.shape
    win_H, win_W = window_sz

    # Blank template for output texture, and indicator matrix indicating which
    # pixels have been generated
    output_texture = np.zeros((output_shape[0], output_shape[1], C))
    has_generated = np.zeros((output_texture.shape[0], output_texture.shape[1]))
    if random_init:
        output_texture[0, 0, :] = np.random.random((C,))
        has_generated[0, 0] = 1
    else:
        output_texture[0:src_H//4, 0:src_W//4, :] = src_texture[0:src_H//4, 0:src_W//4, :]
        has_generated[0:src_H//4, 0:src_W//4] = np.ones((src_H//4, src_W//4))

    # Generate texture row wise
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            if has_generated[i, j] == 0:
                texture_patch = output_texture[max(0, i-win_H):i+1, max(0, j-win_W):j+1, :]
                ValidMask = np.ones((texture_patch.shape[0], texture_patch.shape[1]))
                ValidMask[-1, -1] = 0

                pixels = FindMatches(src_texture, texture_patch, ValidMask)

                output_texture[i, j, :] = pixels[np.random.choice(pixels.shape[0]), :]
                has_generated[i, j] = 1

        if verbose and (i%10==0):
            print('Generating row', i)

    return output_texture


def FindMatches(src_texture, texture_patch, ValidMask, ErrThr=0.1):
    """
    Finds "best" candidate pixels to be generated

    Inputs:
    - src_texture: Source texture image (numpy array)
    - texture_patch: Image patch which closest patch in the source is 
                     to be found
    - ValidMask: Mask indicating which pixels in 'texture_patch' to be 
                 considered, has same height and width as 'texture_patch'
    - ErrThr: Threshold allowed for comparing SSDs to decide candidate pixels

    Output: List of candidate pixels
    """
    src_H, src_W, C = src_texture.shape
    patch_H, patch_W, _ = texture_patch.shape

    dists = compute_SSD(src_texture, texture_patch, ValidMask)
    dists[0, 0:patch_W//2] = 1e8
    dists[0:patch_H//2, 0] = 1e8

    min_dist = np.min(dists)

    i_, j_ = np.nonzero(dists <= (min_dist * (1+ErrThr)))
    pixels = src_texture[i_, j_]

    return pixels


def compute_SSD(src_texture, texture_patch, ValidMask):
    """
    Function computes SSD between texture patch (from generated texture) and
    every possible patch in the source texture image.

    Inputs:
    - src_texture: Source texture image (numpy array)
    - texture_patch: Image patch which closest patch in the source is 
                     to be found
    - ValidMask: Mask indicating which pixels in 'texture_patch' to be 
                 considered, has same height and width as 'texture_patch'

    Output: Matrix of SSDs with dimension as source texture image
    """
    src_H, src_W, C = src_texture.shape
    patch_H, patch_W, _ = texture_patch.shape

    #GaussMask = (np.arange(1, patch_W+1) - patch_W)**2 + ((np.arange(1, patch_H+1) - patch_H)**2).reshape(patch_H, 1)
    #Mask = np.exp(-GaussMask / 100) * ValidMask
    Mask = ValidMask

    src_texture_ = np.pad(src_texture, ((patch_H//2-1+patch_H%2, patch_H//2), (patch_W//2-1+patch_W%2, patch_W//2), (0, 0)), mode = 'constant')
    dists = np.zeros((src_texture_.shape[0], src_texture_.shape[1]))
    for c in range(C):
        D = ndimage.correlate(np.ones((src_texture_.shape[0], src_texture_.shape[1])), Mask*texture_patch[: ,:, c]**2, mode='constant') - 2*ndimage.correlate(src_texture_[:, :, c], Mask*texture_patch[:, :, c], mode='constant') + ndimage.correlate(src_texture_[:, :, c]**2, Mask, mode='constant')
        dists = dists + D

    return dists[0:src_H, 0:src_W] / np.count_nonzero(Mask)
