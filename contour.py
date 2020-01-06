import sys
import json
import argparse
import cv2
import imutils
from imutils import contours
import numpy as np

def largest_from_array(cnts, count):
    largest = []
    for x in range(count):
        max_size = 0
        for i,c in enumerate(cnts):
            size = cv2.contourArea(c)
            if max_size < size:
                max_size = size
                max_index = i
        largest.append(cnts[max_index])
        cnts = np.delete(cnts,max_index,0)
    return largest

def closest_from_array(cnts, area_to_match, count):
    closest = []
    for x in range(count):
        #some large number
        max_area=0xffffffff
        for i,c in enumerate(cnts):
            # print(i,cv2.contourArea(c),cv2.minAreaRect(c))
            size = cv2.contourArea(c)
            diff = abs(area_to_match - size)
            if max_area > diff:
                max_area = diff
                diff_index = i
        closest.append(cnts[diff_index])
        cnts = np.delete(cnts,diff_index,0)
    return closest

def range_from_array(cnts, rangeXtoY):
    inRange = []
    for i,c in enumerate(cnts):
        size = cv2.contourArea(c)
        if size >= rangeXtoY[0] and size <= rangeXtoY[1]:
            inRange.append(c)
    return inRange

def label_contours(image,cnts):
    for i,c in enumerate(cnts):
        try:
            contours.label_contour(image, c, i)
        except:
            print("zero spatial moment used as divisor: label {0} skipped".format(i))
    return image

def contour(image, settings):

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

    contoured_img = image.copy()
    cv2.drawContours(contoured_img, cnts, -1, (0,255,0), 1)    

    return cnts, contoured_img, full_transform

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=
        """
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument("--use_jetson","-j", action="store_true",
                    help= "Use jetson interfaced with picam V2")
    p.add_argument("--resolution","-r", type=str, default="low", 
                    help="Resolution can be max, high, medium, or low.")
    p.add_argument("--show-all","-a", action="store_true", default=False,
                    help="Show all contours in processed image output")
    p.add_argument("--detect-count","-N", type=int, default=1,
                    help="Select N number of contours to detect")
    p.add_argument("--detect-largest","-L", action="store_true", default=False,
                    help="Detect largest of -N number of contour areas")
    p.add_argument("--detect-closest","-C", type=int, default=0,
                    help="Detect closest to area X of -N number of contour areas")
    p.add_argument("--detect-range","-R", type=int, nargs=2,
                    help="Detect all areas of contours in range(-R x y)")

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
        
        cnts, img, trfm = contour(frame, settings)

        if args.show_all:
            out = img.copy()
        else:
            out = frame.copy()

        if args.detect_largest:
            largest = largest_from_array(cnts,args.detect_count)
            out = label_contours(out,largest)

        if args.detect_closest != 0:
            closest = closest_from_array(cnts, args.detect_closest, args.detect_count)
            out = label_contours(out,closest)

        if args.detect_range != None:
            inRange = range_from_array(cnts, args.detect_range)
            out = label_contours(out, inRange)

        dbg = cv2.resize(trfm, (0, 0), None, .25, .25)
        # live_contour = np.concatenate((dbg,img),axis=1)
        cv2.imshow('Contour', out)
        cv2.imshow('Transform',dbg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()

    
