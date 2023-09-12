import cv2
import numpy as np


def detect_black_hsv(img, thresh=90):
    # ideal thresh = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_val = np.array([0,0,0])
    upper_val = np.array([180,255,thresh])
    mask = cv2.inRange(hsv, lower_val, upper_val)
    return mask, hsv
    

def detect_black_gray(img, thresh=90):
    # ideal thresh = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lower_val = np.array([0])
    upper_val = np.array([thresh])
    mask = cv2.inRange(gray, lower_val, upper_val)
    return mask, gray


def closest_white_pixel_to_center(mask, center=[281, 264]):
    # Find the closest white pixel to center
    # Input:
        # mask: binary image
        # center: [row, col] = [281, 264]  # [row, col]
    # Output:
        # closest_point: [row, col]
    white_pixels = np.argwhere(mask == 255)  # position of white pixels [row, col]
    if len(white_pixels) > 0:
        white_pixels_dist = np.subtract(white_pixels, center)  # distance from center
        white_pixels_dist = np.square(white_pixels_dist)  # square distance in x, y
        white_pixels_dist = np.sum(white_pixels_dist, axis=1)  # sum square distance x^2+y^2
        white_pixels_dist = np.sqrt(white_pixels_dist)  # square root to get distance
        white_pixels_min_id = np.argmin(white_pixels_dist)
        closest_white_pixel = white_pixels[white_pixels_min_id]  # [row, col]
        dist = white_pixels_dist[white_pixels_min_id]
    return closest_white_pixel, center, dist


def detect_line_and_error():
    # load video from /raw_video
    vid = cv2.VideoCapture('raw_video/2.mp4')
    # frame by frame loop
    i = 0
    while(vid.isOpened()):
        # read frame
        ret, frame = vid.read()
        # # check if frame is read, if not read next frame
        # if not ret:
        #     continue
        # detect black pixels
        mask, hsv = detect_black_hsv(frame)
        
        # draw all white pixels in green
        for pixel in np.argwhere(mask == 255):
            cv2.circle(frame, tuple(pixel[::-1]), 1, (0,255,0), 1)

        # Detect closest white pixels
        center = [int(mask.shape[0]/2), int(mask.shape[1]/2)]  # [row, col]
        closest_white_pixel, center, dist = closest_white_pixel_to_center(mask, center)

        # draw closest white pixel in green
        cv2.circle(frame, tuple(closest_white_pixel[::-1]), 5, (0,255,0), 2)
        
        # draw center in red
        cv2.circle(frame, tuple(center[::-1]), 5, (0,0,255), 2)
        
        # put dist in frame
        cv2.putText(frame, str(dist), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # show frame
        cv2.imshow('frame', frame)
        # cv2.imshow('mask', mask)

        # # wait for key
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # i += 1

        # press space to continue, q to quit
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord(' '):
            continue


if __name__ == '__main__':
    detect_line_and_error()
    