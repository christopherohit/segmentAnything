import cv2
from sklearn.cluster import KMeans
import numpy as np
from frr import FastReflectionRemoval


def segmentbaseKMeans(kmeans, img, pixel_values):
    labels = kmeans.predict(pixel_values)

    # Reshape the labels back to the original image shape
    labels = labels.reshape((img.shape[0], img.shape[1]))

    k = kmeans.n_clusters
    # Create a mask for each cluster
    mask = np.zeros_like(labels)
    for i in range(k):
        if i == 1:
            mask[labels == i] = 255
        else: mask[labels == i] = 0
    mask_img = mask * (255/k)
    return mask_img


def countDiamond(segment_img, img, threshold=(0, 1000)):
    contours, _ = cv2.findContours(segment_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > threshold[0]) and (area < threshold[1]):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return img

def preprocessing(img):
    # instantiate the algoroithm class
    alg = FastReflectionRemoval(h = 0.2)
    # run the algorithm and get result of shape (H, W, C)
    norm_img = img / 255
    dereflected_img = alg.remove_reflection(norm_img)

    img = np.asarray(dereflected_img * 255, dtype=np.int32)
    img = cv2.convertScaleAbs(img)
    # Convert the image to grayscale

    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # Calculate the unsharp mask
    # unsharp_mask = cv2.addWeighted(gray_img, 15, gray_img, -0.5, 0)

    # Combine the unsharp mask with the original image to create the sharpened image
    # sharpened_img = cv2.cvtColor(cv2.addWeighted(img, 1.5, cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR), -0.5, 0), cv2.COLOR_BGR2RGB)

    blur_img = cv2.medianBlur(img, 3)
    blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

    pixel_values = blur_img.reshape((-1, 1))
    return blur_img, pixel_values


def main():
    # Load image
    img = cv2.imread('data-test/thumbnail_IMG_3808.jpg')

    # Reshape the image to a 2D array of pixels
    blur_img, pixel_values = preprocessing(img)
    cv2.imwrite('blur.jpg', blur_img)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2,init='k-means++', random_state=0).fit(pixel_values)

    mask_img = segmentbaseKMeans(kmeans, img,  pixel_values)
    cv2.imwrite('segment.jpg', mask_img)

    #segment_img = cv2.imread('segment.jpg', 0)
    #_, segment_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_OTSU)
    #count_img = countDiamond(segment_img, img)

    #cv2.imwrite('counted.jpg', count_img)

if __name__ == '__main__':
    main()



