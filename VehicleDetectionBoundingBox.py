
# ### using computer vision to create bounding boxes to locate vehicles on the road. use video provided  which has moving vehicles

import cv2 
from google.colab.patches import cv2_imshow
from skimage.color import rgb2gray
from skimage import filters
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

data_path = "/content/drive/My Drive/Colab Notebooks/CV/"

q = 48  #Higher q size - higher boundingbox size vice-versa.(like threshold to detect object and create bounding box)
frame_ctr,fps,width,height = 0,0,0,0
input_file = data_path + "video1.mp4"
output_file = data_path + "out_video1.mp4"
cap = cv2.VideoCapture(input_file) 
if cap.isOpened(): 
    width  = cap.get(3) 
    height = cap.get(4) 
    fps = cap.get(cv2.CAP_PROP_FPS)  

fourcc = cv2.VideoWriter_fourcc('m','p','4','v') # video formats
out = cv2.VideoWriter(output_file,fourcc , int(fps), (int(width),int(height)))

color = (255, 0, 0) 
success = True
im_old = None

while success:
    success,im = cap.read() 

    if not success:
        break

    if im_old is not None :         
        image = np.copy(im)   
        im = rgb2gray(im)       
        im_old = rgb2gray(im_old)
        frame_diff = cv2.absdiff(im,im_old)        
        frame_diff[frame_diff < frame_diff[np.nonzero(frame_diff)].mean()] = 0
        frame_diff[frame_diff > frame_diff[np.nonzero(frame_diff)].mean()] = 1
        frame_diff = filters.sobel(frame_diff)
        frame_diff = np.stack((frame_diff,)*3, axis=-1)*255      
         
        diff_image = cv2.cvtColor(frame_diff.astype('uint8'), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
        for idx,c in enumerate(contours):
            if hierarchy[0][idx][3] == -1 :
                x,y,w,h = cv2.boundingRect(c)
                if w*h <= q**2:
                    continue
                image = cv2.rectangle(image, (x,y),(x+w,y+h), color, 1)
        if frame_ctr == 2000:
          cv2_imshow(image)
        out.write(image)
    im_old = im
    frame_ctr = frame_ctr + 1
    
    if frame_ctr%30 == 0:
        print("Frames processed :",frame_ctr)
        
cap.release()
out.release()
cv2.destroyAllWindows()
print("Vehicle detection complete.")
