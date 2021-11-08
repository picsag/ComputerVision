import cv2
import numpy as np


def histogramOfGradients(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16  # Number of bins
    bin = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = []
    mag_cells = []
    cellx = celly = 8
    for i in range(0, img.shape[0] // celly):
        for j in range(0, img.shape[1] // cellx):
            bin_cells.append(bin[i * celly: i * celly + celly, j * cellx: j * cellx + cellx])
    mag_cells.append(mag[i * celly: i * celly + celly, j * cellx: j * cellx + cellx])
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    # normalization
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= cv2.norm(hist) + eps
    return hist
