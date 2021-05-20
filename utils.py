import numpy as np
from skimage import feature
from scipy import ndimage
from PIL import Image

# Utilities for image operations, click gaussian and simulations function of humans


def find_center(image_mask):
    """
    First find edges of the mask
    Then takes edge pixels and calculates a center of mask
    :param image_mask:
    :return: returns center array, the list of edge pixels and the dimensions of the mask
    """
    edges = feature.canny(image_mask, sigma=3)
    edges = 1.0 * edges
    list_edge = np.nonzero(edges)
    center = np.array([0, 0])
    center[0] = np.average(list_edge[1])    # Width
    center[1] = np.average(list_edge[0])    # Height
    x_min = np.max(list_edge[1])
    x_max = np.max(list_edge[1])
    y_min = np.max(list_edge[0])
    y_max = np.max(list_edge[0])

    list_edge_dim = np.array([x_min, x_max, y_min, y_max])
    #   print("find_center function used")
    return center, list_edge, list_edge_dim


def make_dist(mask):
    """
    Distance transform of input mask
    :param mask:
    :return: distance transformed image
    """
    #   print("make_dist function used")
    return ndimage.distance_transform_edt(mask)


def make_1d_gaussian(filter, amplitude, center=0, variance=100):
    """
    Generate 1d gaussian filter
    :param filter: Filter
    :param amplitude: Amplitude
    :param center: Center for gaussian (x0 or y0)
    :param variance: Variance
    :return: Gaussian 1d filter
    """
    # print(center)
    return np.sqrt(amplitude) * np.exp(-(filter - center) ** 2 / (2 * variance ** 2))
    # if you want it normalized:
    # return 1/(np.sqrt(2*np.pi*s**2))*np.exp(-(x-m)**2/(2*s**2))


def make_2d_gaussian_filter(center=None, img_w=100, img_h=100, first_click=False):
    """
    Makes a 2D-gaussian filter with the same dimensions
    First click flag increases the amplitude and variance of the gaussian to emphasize importance

    Citation: https://stackoverflow.com/questions/55382361/how-to-generate-a-gaussian-distribution-intensity-for-an-roi
    :param center: center of gaussian
    :param img_w: image width
    :param img_h: image height
    :param first_click: first click flag
    :return: a 2D-gaussian filter
    """
    if center is None:
        center = np.array([50, 50])
    if first_click is False:
        amplitude = 50
        variance = 30
    else:
        amplitude = 100
        variance = 50
    # make two 2D filters of size 100x100
    xx, yy = np.meshgrid(np.arange(img_w), np.arange(img_h))
    # apply gaussian transform to each 2D filters and multiply together
    gaus2d = make_1d_gaussian(filter=xx, amplitude=amplitude, center=center[0], variance=variance) * make_1d_gaussian(filter=yy, amplitude=amplitude, center=center[1], variance=variance)

    return gaus2d


def make_click_image(image_dim, clicks=None, first_click=False):
    """
    Create a gaussian 2D array with same dimensions as the image
    :param image_dim: (1x2) np.array containing [image_height, image_width]
    :param clicks:
    :param first_click:
    :return: gaussian_image which is the same size as image_dim and contains the gaussian values
    """
    gaussian_image = np.zeros((image_dim[0], image_dim[1]), dtype=np.float32)
    if clicks is not None:
        gaussian_filter = make_2d_gaussian_filter(center=clicks, img_h=image_dim[0], img_w=image_dim[1], first_click=first_click)
        gaussian_image = np.add(gaussian_image, gaussian_filter)
        #   print("make_click_image function was used")
        return gaussian_image
    else:
        print("missing click coordinates to make gaussian")
        return False


def normalize_data(data):
    """
    Used for normalizing the data (image) between 0 and 1
    :param data: input data to be normalized
    :return: normalized data
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def compare_mask(true_mask, pred_mask):
    """
    Calculates true foreground, true background, false foreground and false background
    from the true mask compared to predicted mask
    :param true_mask: ground truth
    :param pred_mask: prediction
    :return:
    """
    # calculate inverse of true mask, used for calculations
    inverse_true_mask = 1 - true_mask
    # correct foreground predicted pixels
    true_foreground_mask = np.multiply(true_mask, pred_mask)
    # false background predicted pixels also known as missing foreground pixels
    false_background_mask = np.multiply((1 - true_foreground_mask), true_mask)
    # false foreground predicted pixels also known as missing background pixels
    false_foreground_mask = np.multiply(inverse_true_mask, pred_mask)
    # correct background predicted pixels
    true_background_mask = np.multiply((1 - pred_mask), inverse_true_mask)

    return true_foreground_mask, false_background_mask, false_foreground_mask, true_background_mask


def sim_first_click(true_mask):
    """
    Calculates the center of the true_mask in order to simulated a first click from a human
    Then calls the make_click function to generate a gaussian filter around the center.
    :param true_mask:
    :return: gaussian filter around the center of the true_mask
    """
    image_dim = np.array([true_mask.shape[0], true_mask.shape[1]])
    # Click center
    center_true_mask, _, __ = find_center(true_mask)  # maybe change to make_dist in case of edge cases
    gaussian_first_click_center = make_click_image(image_dim=image_dim, clicks=center_true_mask, first_click=True)

    return gaussian_first_click_center, center_true_mask


def sim_pos_click(false_background_mask):
    """
    Takes the false_background_mask and calculates a distance map on it
    Then find the largest value (the center) and calls the make_click function to get a gaussian filter on the center
    :param false_background_mask: Missing foreground mask after prediction
    :return: gaussian filter around the center of false_background_mask
    """
    image_dim = np.array([false_background_mask.shape[0], false_background_mask.shape[1]])
    # make distance transforms of the false_background_mask in order to click next positive position
    # false_background_mask is also the missing foreground
    pos_dist_map = make_dist(false_background_mask)
    _temp = np.flip(np.transpose(np.asarray(np.where(pos_dist_map == pos_dist_map.max()))))  # reversed order of x,y

    pos_click_index = _temp[0]
    pos_gaussian = make_click_image(image_dim=image_dim, clicks=pos_click_index, first_click=False)

    return pos_gaussian


def sim_neg_click(false_foreground_mask, center):
    """
    Takes the false_foreground_mask and calculates the edges of it
    Then calculates the euclidean distance from each edge pixel to the center of the truth_mask
    The lowest distance pixel is then used to generate a negative click on the edge of the true mask
    the reason for needed the minimum distance of towards the center of the mask is to know which side
    of the false_foreground_mask to generate the click.
    With calling the make_click function to get a gaussian filter on the edge/lowest distance point
    :param false_foreground_mask:
    :param center: center of the true mask in order to calculate the correct edge side to make the negative click
    :return: gaussian filter around the center of false_foreground_mask
    """
    image_dim = np.array([false_foreground_mask.shape[0], false_foreground_mask.shape[1]])
    neg_dist_map = np.nonzero(feature.canny(false_foreground_mask, sigma=1))
    min_dist = np.inf
    neg_click_index = None
    print(len(neg_dist_map[0]))
    for i in range(len(neg_dist_map[0])):
        point = np.transpose(np.array([[neg_dist_map[1][i]], [neg_dist_map[0][i]]]))
        dist = np.linalg.norm(center - point)
        if dist < min_dist:
            min_dist = dist
            neg_click_index = point[0]

    neg_gaussian = make_click_image(image_dim=image_dim, clicks=neg_click_index, first_click=False)

    return neg_gaussian


def make_test_images():
    """
    functions load the test image and test mask
    normalizes the image
    changes the mask to be between 0 and 1 instead of 255
    :return:
    """
    org_image = np.array(Image.open('tester.jpg').convert("RGB"), dtype=np.float32)
    org_image = normalize_data(org_image)
    true_mask = np.array(Image.open('tester_mask.gif').convert("L"), dtype=np.float32)
    true_mask[true_mask == 255.0] = 1.0
    true_mask[true_mask <= 0.0] = 0.0

    return org_image, true_mask
