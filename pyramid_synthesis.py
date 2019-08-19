import numpy as np
from scipy import stats
from scipy import ndimage
import cv2

def make_cdf(img, n_bins=256):
    """
    Creates CDF of input image (map from [0,255] to [0,1])

    Inputs:
    - img: input image
    - n_bins: Number of bins used

    Output:
    - Dictionary containing Cummulative frequencies (CDF) of pixel values, 
      contained in array, and number of items (pixels) used to compute CDF
    """
    cdf = stats.cumfreq(img, n_bins, (0, 255))[0]
    cdf_ = {'cdf': np.array(cdf) / int(max(cdf)), 'n_items': int(max(cdf))}

    return cdf_


def make_inv_cdf(cdf):
    """
    Creates inverse CDF of input CDF (map from [0,1] to [0,255])

    Input:
    - cdf: Dictionary containing CDF whose inverse is to be computed, and
           number of items (pixels) used to compute that CDF
    Output:
    - Inverse CDF as a array
    """
    inv_cdf = np.zeros((cdf['n_items'] + 1))
    cdf_ = np.int64(cdf['cdf'] * cdf['n_items'])

    idx, vals = np.unique(cdf_, return_index=True)
 
    for i in range(len(idx)-1):
        inv_cdf[idx[i]:idx[i+1]] = vals[i] * np.ones(len(inv_cdf[idx[i]:idx[i+1]]))
    inv_cdf[idx[i+1]] = vals[i+1]
 
    return {'inv_cdf': inv_cdf, 'n_items': cdf['n_items']}


def lookup(f, val, desc):
    """
    Lookup value from given function

    Inputs:
    - f: CDF or Inverse-CDF to lookup
    - val: Value to lookup
    - desc: Either 'cdf' or 'inv_cdf', indicating the type of CDF to lookup

    Output:
    - Return the value of function at val
    """
    if desc=='cdf':
        return f['cdf'][int(val.round())]

    if desc=='inv_cdf':
        return f['inv_cdf'][int((val * f['n_items']).round())]


def match_histogram(img1, img2):
    """
    Modifies image 1 to have same histogram as image 2
    """
    img1_cdf = make_cdf(img1)
    img2_cdf = make_cdf(img2)

    img2_inv_cdf = make_inv_cdf(img2_cdf)

    H, W = img1.shape
    img1 = img1.reshape(H*W)
    for i in range(H*W):
        img1[i] = lookup(img2_inv_cdf,
                         lookup(img1_cdf, img1[i], 'cdf'),
                         'inv_cdf')

    return img1.reshape(H, W)


def reduce(img):
    """
    Reduce operation on the input image: Apply filter then downsample
    """
    lp_filter = [1/16, 4/16, 6/16, 4/16, 1/16]

    filtered_img = ndimage.correlate1d(img, lp_filter, axis=0)
    filtered_img = ndimage.correlate1d(filtered_img, lp_filter, axis=1)

    H, W = img.shape

    return filtered_img[range(1, W, 2), :][:, range(1, H, 2)]


def expand(img):
    """
    Expand operation on the input image: Upsample, then apply filter
    -- Not implemented
    """
    pass

def make_pyramid(img):
    """
    Constructs laplacian image pyramid with given input image

    Input: Image whose pyramid to be constructed
    Output: Image pyramid represented as a list
    """
    img_ = img
    img_pyramid = []
    for l in range(np.int(np.log(min((img.shape[0],img.shape[1])))/np.log(2))):
        lp = cv2.pyrDown(img_)
        img_pyramid.append({'lp': lp, 
                            'bp': cv2.subtract(img_, cv2.pyrUp(lp))})
        img_ = lp

    return img_pyramid


def collapse_pyramid(img_pyramid):
    """
    Reconstructs image from input image pyramid

    Input: Image pyramid
    Output: Reconstructed image from pyramid
    """
    img = img_pyramid[-1]['bp'] + cv2.pyrUp(img_pyramid[-1]['lp'])
    for l in range(len(img_pyramid)-2, -1, -1):
        img = img_pyramid[l]['bp'] + cv2.pyrUp(img)

    return img


def match_texture(img, src_texture, n_iter=120, verbose=True):
    """
    Synthesize texture image

    Inputs:
    - img: Randomized image where the generated image is to be placed, of
           shape (H, W, C)
    - src_texture: Sample texture image, of shape (H, W, C)
    - n_iter: Number of iterations to use
    - verbose: If True, print progress to console

    Output: Generated texture image of same shape as 'img'
    """
    # Clipping sizes of source and output texture images
    H, W, C = src_texture.shape
    src_texture = src_texture[0:2**int(np.floor(np.log(H)/np.log(2))),
                              0:2**int(np.floor(np.log(W)/np.log(2))), :]
    H, W, C = img.shape
    img = img[0:2**int(np.floor(np.log(H)/np.log(2))),
              0:2**int(np.floor(np.log(W)/np.log(2))), :]

    # Remove correlation between colour channels
    if C>1:
        mean_clr, M, src_texture = transform_img(src_texture)

    # Texture synthesis
    for c in range(C):
        img[:,:,c] = match_histogram(img[:,:,c], src_texture[:,:,c])
        analysis_pyramid = make_pyramid(src_texture[:,:,c])

        for it in range(n_iter):
            synthesis_pyramid = make_pyramid(img[:,:,c])

            for l in range(len(synthesis_pyramid)):
                    synthesis_pyramid[l]['lp'] = match_histogram(synthesis_pyramid[l]['lp'], analysis_pyramid[l]['lp'])
                    synthesis_pyramid[l]['bp'] = match_histogram(synthesis_pyramid[l]['bp'], analysis_pyramid[l]['bp']) 

            img[:,:,c] = collapse_pyramid(synthesis_pyramid)
            img[:,:,c] = match_histogram(img[:,:,c], src_texture[:,:,c])

            if verbose:
                print('Iteration', it)

    if C>1:
        img = inv_transform_img(img, mean_clr, M)

    return img


def transform_img(img):
    """
    Transforms input image by removing correlations between colour channels.
    When called, 'img' should be the souce texture image.

    Input:
    - img: Input image of shape (H, W, C)
    Output:
    - (mean_clr, M, t_img): Tuple containing the mean colour, the decorrelation
                            matrix, and the transformed image
    """
    H, W, C = img.shape
    img = img.reshape(H*W, C).T # Shape (C, H*W)

    mean_colour = np.mean(img, axis=-1, keepdims=True)
    img = img - mean_colour

    U, S, V = np.linalg.svd(img, full_matrices=False)

    # M is the decorrelating matrix
    M = np.dot(np.diag(np.sqrt(S)), U)

    img = np.dot(M, img)

    return mean_colour, M, (img.T).reshape(H, W, C)


def inv_transform_img(img, mean_colour, M):
    """
    Inverse transform of input image.
    When called, 'img' should be the generated texture image

    Inputs:
    - img: Input image of shape (H, W, C)
    - mean_colour: Mean colour of shape (C, 1)
    - M: Decorrelation matrix, of shape (C, C)

    Output: Inverse transformed image, same shape as input image
    """
    H, W, C = img.shape
    img = img.reshape(H*W, C).T # Shape(C, H*W)

    img = np.dot(M, img)
    img += mean_colour

    return (img.T).reshape(H, W, C)
