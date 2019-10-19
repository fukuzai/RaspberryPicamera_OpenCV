import time
import picamera
from datetime import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

#HOW TO INSTALL matplotlib on your raspberry pi
#sudo apt-get install python-matplotlib

with picamera.PiCamera() as camera:

    #set camera resolutin
    camera.resolution = (1024, 768)
    camera.start_preview()
    time.sleep(2)
    timestr = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    fn = timestr + '.jpg'
    #save image
    camera.capture(fn)

    
#load image
img = cv2.imread(fn)

#convert
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fn_gray = timestr + '_gray.jpg'
cv2.imwrite(fn_gray, grayimg)

#threshold
#ret, bin = cv2.threshold(grayimg, 0, 40, cv2.THRESH_BINARY)
ret, bin = cv2.threshold(grayimg,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
fn_bin = timestr + '_bin.jpg'
cv2.imwrite(fn_bin, bin)

#invert ( necessary if cv2.THRESH_BINARY_INV is used)
#bin_inv = cv2.bitwise_not(bin)
#fn_bininv = timestr + '_bininv.jpg'
#cv2.imwrite(fn_bininv, bin_inv)

#find contours of white blobs in the bin image
contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

blob_and_contours = np.copy(bin)
min_area = 60
large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
cv2.drawContours(blob_and_contours, large_contours, -1, (255,0,0))

#output
print('number of blobs : %d' % len(large_contours))
plt.imshow(blob_and_contours)
fn_contour= timestr + '_blob_and_contours.jpg'
plt.savefig(fn_contour)
