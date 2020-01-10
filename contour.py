import sys
import json
import argparse
import cv2
import math
import imutils
from imutils import contours
import numpy as np

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def euclidean_distance(point1,point2):
    x1,y1 = point1
    x2,y2 = point2
    return math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))

def center_of_contour(contour):
    m = cv2.moments(contour)
    if m["m00"] != 0:
        cX = int(m["m10"] / m["m00"])
        cY = int(m["m01"] / m["m00"])
        return [cX,cY]
    else:
        return [0,0]

def valid_seperation(cnts, target, seperation):
    #if first contour, return
    if len(cnts) == 0:
        return True
    c1 = center_of_contour(target)
    #need to vet each potential target for incomplete segmentation
    if c1 == [0,0]:
        return False
    for i,c in enumerate(cnts):
        c2 = center_of_contour(c)
        dist = euclidean_distance(c1,c2)
        if dist < seperation:
            return False
    return True

def largest_from_array(cnts, count, seperation):
    largest = []
    attempts = 0
    while len(largest) < count and attempts < 10:
        max_size = 0
        for i,c in enumerate(cnts):
            size = cv2.contourArea(c)
            if max_size < size:
                max_size = size
                max_index = i
        if valid_seperation(largest, cnts[max_index], seperation):
            largest.append(cnts[max_index])
        #else don't append and delete contour
        cnts = np.delete(cnts,max_index,0)
        attempts += 1
    return largest

def closest_from_array(cnts, area_to_match, count, seperation):
    closest = []
    attempts = 0
    while len(closest) < count and attempts < 10:
        #some large number
        max_area=0xffffffff
        for i,c in enumerate(cnts):
            # print(i,cv2.contourArea(c),cv2.minAreaRect(c))
            size = cv2.contourArea(c)
            diff = abs(area_to_match - size)
            if max_area > diff:
                max_area = diff
                diff_index = i
        if valid_seperation(closest, cnts[diff_index], seperation):
            closest.append(cnts[diff_index])
        #else don't append and delete contour
        cnts = np.delete(cnts,diff_index,0)
        attempts += 1
    return closest

def range_from_array(cnts, rangeXtoY, seperation):
    inRange = []
    for i,c in enumerate(cnts):
        size = cv2.contourArea(c)
        if size >= rangeXtoY[0] and size <= rangeXtoY[1]:
            if valid_seperation(inRange, c, seperation):
                inRange.append(c)
    return inRange

def label_contours(image,cnts,show_sizes=False):
    for i,c in enumerate(cnts):
        try:
            contours.label_contour(image, c, i)
            if show_sizes:
                size = cv2.contourArea(c)
                cv2.putText(image, "Size #{0}:{1}".format(i,size),
                    (10,80+30*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        except:
            print("contour not segmented correctly: label {0} skipped".format(i))
    return image

def bound_contour(image, contour):
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    tl = (x,y)
    tr = (x+w,y)
    bl = (x,y+h)
    br = (x+w,y+h)

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 2)
    cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints
    # dA is Y direction, if not rotating
    dA = euclidean_distance((tltrX, tltrY), (blbrX, blbrY))
    dB = euclidean_distance((tlblX, tlblY), (trbrX, trbrY))

    cv2.putText(image, "{:.2f}px".format(dA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
    cv2.putText(image, "{:.2f}px".format(dB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
    
    return image, dA, dB

def calc_distance_from_lens(image, pixel_array):
    global pixel_conversion_factor
    global resolution_factor
    global object_actual_dimension
    global res

    for i,p in enumerate(pixel_array):
        height = pixel_conversion_factor - (object_actual_dimension/p)
        height = height * resolution_factor
        cv2.putText(image, "Height#{0}: {1:0.2f}cm".format(i,height),
                    (int(res[0])-200,80+30*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

    return image

def measure_contours(image,cnts, bounding_box=False, measure_dist=False):
    if bounding_box:
        measY = []
        for i,c in enumerate(cnts):
            image, dY, _ = bound_contour(image, c)
            measY.append(dY)
        
        if measure_dist:
            image = calc_distance_from_lens(image, measY)
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
    
    p.add_argument("--use-laptop","-l", action="store_true", default=False,
                    help= "Use laptop camera")
    p.add_argument("--reset-config", action="store_true", default=False,
                    help="Reset contour config options using .original file")
    p.add_argument("--save-config", type=str, default=None,
                    help="Save .contour.json file to designated name file to be used later")
    p.add_argument("--use-config", type=str, default=None,
                    help="Use test settings for designated name file")
    p.add_argument("--snap-shot","-ss", action="store_true", default=False,
                    help="Take a snap shot and save it to file")
    p.add_argument("--show-transform","-t", action="store_true", default=False,
                    help="Show image transform in seperate window")
    p.add_argument("--resolution","-r", type=str, default="low", 
                    help="Resolution can be max, high, medium, or low.")
    p.add_argument("--show-all","-a", action="store_true", default=False,
                    help="Show all contours in processed image output")
    p.add_argument("--show-size","-s", action="store_true", default=False,
                    help="Show sizes of detected contours")
    p.add_argument("--detect-count","-N", type=int, default=1,
                    help="Select N number of contours to detect")
    p.add_argument("--detect-largest","-L", action="store_true", default=False,
                    help="Detect largest of -N number of contour areas")
    p.add_argument("--detect-closest","-C", type=int, default=0,
                    help="Detect closest to area X of -N number of contour areas")
    p.add_argument("--detect-range","-R", type=int, nargs=2,
                    help="Detect all areas of contours in range(-R x y)")
    p.add_argument("--pixel_seperation","-p", type=int, default=0,
                    help="May need to seperate detected contour centers by pixel")
    p.add_argument("--bounding-box","-b", action="store_true", default=False,
                    help="Draw non-rotating bounding box around detected contours")
    p.add_argument("--measure-from-lens","-m", action="store_true", default=False,
                    help="Measure distance of contour from lens using y dimension, given lens measurements")

    return(p.parse_args())

if __name__ == '__main__':
    from camera import *

    if sys.version_info<(3,0,0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)
        
    args = cmdline_args()

    if args.use_config:
        settings_file = args.use_config+".json"
    else:
        settings_file = ".contour.json"

    if args.reset_config:
        with open(".original", 'r') as f:
            settings = json.load(f)
        with open(settings_file, 'w') as f:
            json.dump(settings,f, indent=4)
        exit(0)
    else:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        print(settings)
        if args.save_config and args.use_config == None:
            with open(args.save_config+".json", 'w') as f:
                json.dump(settings,f,indent=4)
            exit(0)
        

    res = resolutions[args.resolution]
    if args.use_laptop == False:
        cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=2,display_width=res[0],display_height=res[1]), cv2.CAP_GSTREAMER)
    else:
        cam = cv2.VideoCapture(0)

    if args.measure_from_lens:
        object_actual_dimension = float(input("Enter measurement of object to be measured in y dimension(cm): "))
        if settings["pixelConversionFactor"] == "NeedCalibratedValue":
            pixel_conversion_factor = float(input("Enter pixel conversion factor(cm/px): "))
        else:
            pixel_conversion_factor = settings["pixelConversionFactor"]
        resolution_factor = calc_resolution_factor(res[1])

    seperation = args.pixel_seperation

    cv2.namedWindow('Contour', cv2.WINDOW_AUTOSIZE)
    if args.show_transform:
        cv2.namedWindow('Transform',cv2.WINDOW_AUTOSIZE)

    while(True):
        # Capture frame-by-frame
        ret, frame = cam.read()
        
        cnts, img, trfm = contour(frame, settings)

        if args.show_all:
            out = img.copy()
        else:
            out = frame.copy()

        if args.detect_largest:
            largest = largest_from_array(cnts,args.detect_count, seperation)
            out = label_contours(out,largest, args.show_size)
            out = measure_contours(out,largest, args.bounding_box, args.measure_from_lens)

        if args.detect_closest != 0:
            closest = closest_from_array(cnts, args.detect_closest, args.detect_count, seperation)
            out = label_contours(out,closest, args.show_size)
            out = measure_contours(out,closest, args.bounding_box, args.measure_from_lens)

        if args.detect_range != None:
            inRange = range_from_array(cnts, args.detect_range, seperation)
            out = label_contours(out, inRange, args.show_size)
            out = measure_contours(out,inRange, args.bounding_box, args.measure_from_lens)

        h,w,_ = out.shape
        cv2.putText(out, "Res:{0}x{1}".format(w,h),
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)

        if args.snap_shot:
            cv2.imwrite('out.jpg',out)
            cv2.imwrite('trfm.jpg',trfm)
            break
        else:
            dbg = cv2.resize(trfm, (0, 0), None, .30, .30)
            # live_contour = np.concatenate((dbg,img),axis=1)
            cv2.imshow('Contour', out)
            if args.show_transform:
                window_alignment = int(res[0])+100
                cv2.imshow('Transform',dbg)
                cv2.moveWindow('Transform',window_alignment,0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()

    
