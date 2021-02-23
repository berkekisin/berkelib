import numpy as np
import cv2

image_location = r"C:\Users\kisin\Desktop\Cut.png"
graphh = r"C:\Users\kisin\Desktop\graph2.png"


def gauss(kernel, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-kernel ** 2 / (2 * sigma ** 2))


def gaussian_derivative(kernel, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- (kernel ** 2) / (2 * sigma ** 2)) * (
            - (2 * kernel) / (2 * sigma ** 2))


def gaussian_filter(image, sigma):
    k_half = 3 * sigma
    kernel = np.linspace(-k_half, k_half, 2 * k_half + 1)
    kernel = gauss(kernel, sigma)
    image = cv2.copyMakeBorder(image, k_half, k_half, k_half, k_half, cv2.BORDER_CONSTANT)


    image_temp = np.empty_like(image)
    image_filtered = np.empty_like(image)
    height = image.shape[0]
    width = image.shape[1]

    for i in range(height):
        for j in range(k_half, width - k_half):
            image_temp[i, j, 0] = np.sum(kernel * image[i, j - k_half: j + k_half + 1, 0])
            image_temp[i, j, 1] = np.sum(kernel * image[i, j - k_half: j + k_half + 1, 1])
            image_temp[i, j, 2] = np.sum(kernel * image[i, j - k_half: j + k_half + 1, 2])

    for i in range(k_half, height - k_half):
        for j in range(width):
            image_filtered[i, j, 0] = np.sum(kernel * image_temp[i - k_half:i + k_half + 1, j, 0])
            image_filtered[i, j, 1] = np.sum(kernel * image_temp[i - k_half:i + k_half + 1, j, 1])
            image_filtered[i, j, 2] = np.sum(kernel * image_temp[i - k_half:i + k_half + 1, j, 2])
            # image_filtered[i, j] = np.sum(kernel * image_temp[i - k_half:i + k_half + 1, j])

    return image_filtered[k_half:height - k_half, k_half:width - k_half]


def gaussian_derivatives(image, sigma):
    k_half = 3 * sigma
    kernel = np.linspace(-k_half, k_half, 2 * k_half + 1)
    kernel = gaussian_derivative(kernel, sigma)
    image = cv2.copyMakeBorder(image, k_half, k_half, k_half, k_half, cv2.BORDER_CONSTANT)

    # create image to restore row filter
    image_filtered_x = np.empty_like(image)
    image_filtered_y = np.empty_like(image)

    height = image.shape[0]
    width = image.shape[1]

    for i in range(height):
        for j in range(k_half, width - k_half):
            image_filtered_x[i, j, 0] = np.sum(kernel * image[i, j - k_half: j + k_half + 1, 0])
            image_filtered_x[i, j, 1] = np.sum(kernel * image[i, j - k_half: j + k_half + 1, 1])
            image_filtered_x[i, j, 2] = np.sum(kernel * image[i, j - k_half: j + k_half + 1, 2])

    for i in range(k_half, height - k_half):
        for j in range(width):
            image_filtered_y[i, j, 0] = np.sum(kernel * image[i - k_half:i + k_half + 1, j, 0])
            image_filtered_y[i, j, 1] = np.sum(kernel * image[i - k_half:i + k_half + 1, j, 1])
            image_filtered_y[i, j, 2] = np.sum(kernel * image[i - k_half:i + k_half + 1, j, 2])

    return image_filtered_x[k_half:height - k_half,
           k_half:width - k_half], image_filtered_y[k_half:height - k_half,
                                   k_half:width - k_half]


def detectGraph(x, y):
    img = np.empty_like(x)

    height = img.shape[0]
    width = img.shape[1]

    for i in range(height):
        for j in range(width):
            img[i, j] = (x[i, j] + y[i, j])

    return img


img = cv2.imread(graphh)
#laplacian = cv2.Laplacian(img,cv2.CV_64F)
#img = gaussian_filter(img, 2)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x, y = gaussian_derivatives(img, 1)
a = detectGraph(x,y)
# cv2.imshow("derivative_y", img)
cv2.imwrite(r"C:\Users\kisin\Desktop\meric3.jpg", a)
cv2.imshow("derivative_x", a)
cv2.waitKey(0)
