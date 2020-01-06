import sys
import json
import argparse
import cv2
import imutils
from imutils import contours
import numpy as np

def contour(image, settings, detect_largest=1):

    im = image.copy()
    if settings['blurImage'] == True:
        im = cv2.GaussianBlur(im, settings['blurKernelSize'], 0)

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    lt = settings['edgingLowThresh']
    ht = settings['edgingHighThresh']
    
    edges = cv2.Canny(gray, lt[0], lt[1])
    edges_high_thresh = cv2.Canny(gray, ht[0], ht[1])

    #first stage of transform
    edging = np.hstack((gray, edges, edges_high_thresh))

    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    edges_filtered = cv2.Canny(gray_filtered, 60, 120)

    #extra filtering to be considered later
    filtering = np.hstack((gray, gray_filtered, edges_filtered))
    
    #choose which transform to use
    transformType=[edges,edges_high_thresh,edges_filtered]
    transform = transformType[settings['selectTransform']]

    dilate = cv2.dilate(transform, None, iterations=settings["dilateIteration"])
    erode = cv2.erode(dilate, None, iterations=settings["erodeIteration"])

    defining = np.hstack((transform, dilate, erode))
    full_transform = np.vstack((edging, filtering, defining))
    full_transform = np.concatenate((edging, filtering, defining), axis=0)

    #use last transformation
    transform = erode

    cnts = cv2.findContours(transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)

    tmp = cnts
    for i in range(detect_largest):
        cnt, index = largest_from_array(tmp)
        #must delete from array for next round
        tmp = np.delete(tmp,index,0)
        cv2.drawContours(image, tmp, -1, (0,255,0), 3)    

    cv2.drawContours(image, cnts, -1, (0,255,0), 1)    

    return cnts, image, full_transform

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=
        """
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument('--use_jetson',action="store_true",
                    help= "Use jetson interfaced with picam V2")
    p.add_argument("--resolution","-r", type=str, default="low", 
                    help="Resolution can be max, high, medium, or low.")
    p.add_argument("--detect-largest","-L", type=int, default=1
                    help="Detect largest N number of contours")

    return(p.parse_args())

if __name__ == '__main__':
    from camera import *

    if sys.version_info<(3,0,0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)
        
    args = cmdline_args()

    settings_file = ".contour.json"
    with open(settings_file, 'r') as f:
        settings = json.load(f)

    res = resolutions[args.resolution]
    if args.use_jetson == True:
        cam = cv2.VideoCapture(gstreamer_pipeline(display_width=res[0],display_height=res[1]))
    else:
        cam = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cam.read()
        
        img,trfm = contour(frame, settings, args.detect_largest)

        dbg = cv2.resize(dbg, (0, 0), None, .25, .25)
        # live_contour = np.concatenate((dbg,img),axis=1)
        cv2.imshow('Contour', img)
        cv2.imshow('Transform',dbg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()

    
