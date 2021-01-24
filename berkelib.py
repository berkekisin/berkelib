import numpy as np
import cv2

image_location = ""

def gauss(kernel, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-kernel ** 2 / (2 * sigma ** 2))


def gaussian_derivative(kernel, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- (kernel ** 2) / (2 * sigma ** 2)) * (- (2 * kernel) / (2 * sigma ** 2))


def gaussian_filter(image, sigma):
    # generate kernel
    k_length_half = 3 * sigma
    kernel = np.linspace(-k_length_half, k_length_half, 2 * k_length_half + 1)
    kernel = gauss(kernel, sigma)
    kernel = np.expand_dims(kernel, -1)

    # add padding
    image = cv2.copyMakeBorder(image, k_length_half, k_length_half, k_length_half, k_length_half, cv2.BORDER_CONSTANT)

    # create image to restore row filter
    image_temp = np.empty_like(image)
    image_filtered = np.empty_like(image)

    height = image.shape[0]
    width = image.shape[1]

    for i in range(height):
        for j in range(k_length_half, width - k_length_half):
            image_temp[i, j] = np.sum(kernel * image[i, j - k_length_half: j + k_length_half + 1], axis=0)

    for i in range(k_length_half, height - k_length_half):
        for j in range(width):
            image_filtered[i, j] = np.sum(kernel * image_temp[i - k_length_half:i + k_length_half + 1, j], axis=0)

    return image_filtered[k_length_half:height - k_length_half, k_length_half:width - k_length_half]


def gaussian_derivatives(image, sigma):
    # generate kernel
    k_length_half = 3 * sigma
    kernel = np.linspace(-k_length_half, k_length_half, 2 * k_length_half + 1)
    kernel = gaussian_derivative(kernel, sigma)

    # add padding
    image = cv2.copyMakeBorder(image, k_length_half, k_length_half, k_length_half, k_length_half, cv2.BORDER_CONSTANT)

    # create image to restore row filter
    image_filtered_x = np.empty_like(image)
    image_filtered_y = np.empty_like(image)

    height = image.shape[0]
    width = image.shape[1]

    for i in range(height):
        for j in range(k_length_half, width - k_length_half):
            image_filtered_x[i, j] = np.sum(kernel * image[i, j - k_length_half: j + k_length_half + 1], axis=0)

    for i in range(k_length_half, height - k_length_half):
        for j in range(width):
            image_filtered_y[i, j] = np.sum(kernel * image[i - k_length_half:i + k_length_half + 1, j], axis=0)

    return image_filtered_x[k_length_half:height - k_length_half,
           k_length_half:width - k_length_half], image_filtered_y[k_length_half:height - k_length_half,
                                                 k_length_half:width - k_length_half]


img = cv2.imread(image_location)
img = gaussian_filter(img, 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x, y = gaussian_derivatives(img, 2)
cv2.imshow("derivative_y", y)
# cv2.imwrite(r"C:\Users\kisin\Desktop\meric3.jpg", y)
cv2.imshow("derivative_x", x)
cv2.waitKey(0)
