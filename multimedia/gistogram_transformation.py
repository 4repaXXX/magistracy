import cv2

def linear(path):
    image = cv2.imread(path)
    a = 1.5 # коэффициент контрастности
    b = 50  # яркость

    adjusted = cv2.convertScaleAbs(image, alpha=a, beta=b)

    return adjusted

def threshold():
    import cv2

    image = cv2.imread('th.jpeg', cv2.IMREAD_GRAYSCALE)

    ret, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    return threshold

def thershold_adaptive():
    import cv2

    image = cv2.imread('th.jpeg', cv2.IMREAD_GRAYSCALE)

    adaptive_thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    return adaptive_thresholded




def main():
    thershold_adaptive()
    
if __name__ == "__main__":
    main()
