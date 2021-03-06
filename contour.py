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
    if len(cnts) < count:
        count = len(cnts)
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
        if len(cnts) == 0:
            break
    return largest

def closest_from_array(cnts, area_to_match, count, seperation):
    closest = []
    attempts = 0
    if len(cnts) < count:
        count = len(cnts)
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

def range_from_array(cnts, rangeXtoY):
    in_range = []
    for i,c in enumerate(cnts):
        size = cv2.contourArea(c)
        if size >= rangeXtoY[0] and size <= rangeXtoY[1]:
            # if valid_seperation(inRange, c, seperation):
            in_range.append(c)
    return in_range

def label_contours(image,cnts,show_sizes=False,measure=False):
    for i,c in enumerate(cnts):
        if measure:
            bound_contour(image, c, measure)
        else:
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

def bound_contour(image, contour, measure=False):
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)

    if measure == False:
        return image, 0, 0

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
                    (int(res[0])-250,20+30*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

    return image

def measure_contours(image,cnts, bounding_box=False, measure_dist=False):
    dY = 0
    if bounding_box:
        measY = []
        for i,c in enumerate(cnts):
            image, dY, dX = bound_contour(image, c, measure_dist)
            measY.append(dY)
        
        if measure_dist:
            image = calc_distance_from_lens(image, measY)
    return image, dY

def contour(image, settings):

    im = image.copy()
    if settings['blurImage'] == True:
        im = cv2.GaussianBlur(im, (7,7), 0)

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

    rt = settings["contourHierarchy"]
    if rt == "EXTERNAL":
        rt = 0
    else:
        rt = 2

    cnts = cv2.findContours(transform, rt, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    try:
        (cnts, _) = contours.sort_contours(cnts)
    except:
        print("No contours detectd")

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
    p.add_argument("--reset-config","-rc", action="store_true", default=False,
                    help="Reset contour config options using .original file")
    p.add_argument("--save-config","-sc", type=str, default=None,
                    help="Save .contour.json file to designated name file to be used later")
    p.add_argument("--use-config","-uc", type=str, default=None,
                    help="Use test settings for designated name file")
    p.add_argument("--load-config","-lc", action="store_true", default=False,
                    help="load use-config test file into default .contour.json file")
    p.add_argument("--snap-shot","-ss", action="store_true", default=False,
                    help="Take a snap shot and save it to file")
    p.add_argument("--show-transform","-t", action="store_true", default=False,
                    help="Show image transform in seperate window")
    p.add_argument("--resolution","-r", type=str, default="low", 
                    help="Resolution can be max, high, medium, or low.")
    p.add_argument("--show-all","-a", action="store_true", default=False,
                    help="Show all contours in processed image output")
    p.add_argument("--label","-lb", action="store_true", default=False,
                    help="Label contours with thicker lines and number them")
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
    p.add_argument("--set-pixel-factor", "-spf", action="store_true", default=False,
                    help="Set the pixel conversion factor in .contour.json")
    p.add_argument("--detect-blob","-B", action="store_true", default=False,
                    help="Detect blobs ontop of other detection algorithms")
    p.add_argument("--select-roi","-roi", action="store_true", default=False,
                    help="Select roi and save to test settings")
    # p.add_argument("--crop-height","-ch", type=int, default=0,
    #                 help="Crop image height by number of pixels")
    # p.add_argument("--crop-width","-cw", type=int, default=0,
    #                 help="Crop image width by number of pixels")

    return(p.parse_args())

if __name__ == '__main__':
    from camera import *
    from blob import *

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
        elif args.load_config and args.use_config:
            with open(".contour.json", 'w') as f:
                json.dump(settings,f,indent=4)
        

    res = resolutions[args.resolution]
    if args.use_laptop == False:
        cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=2,display_width=res[0],display_height=res[1]), cv2.CAP_GSTREAMER)
    else:
        cam = cv2.VideoCapture(1) 
        if cam is None or not cam.isOpened():
            cam = cv2.VideoCapture(0)

    measure = settings["measure"] == True or args.measure_from_lens == True
    if measure:
        if settings["itemDimension"] == "None": 
            object_actual_dimension = float(input("Enter measurement of object to be measured in y dimension(cm): "))
        else:
            object_actual_dimension = float(settings["itemDimension"])
        if settings["pixelConversionFactor"] == "NeedCalibratedValue":
            pixel_conversion_factor = float(input("Enter pixel conversion factor(cm/px): "))
        else:
            pixel_conversion_factor = settings["pixelConversionFactor"]
        #if settings["cropImage"] == True:
        #    r = settings["cropDimensions"]
        #    yDim = r[3]
        #    resolution_factor = calc_resolution_factor(yDim)
        #else:
        resolution_factor = calc_resolution_factor(res[1])

    seperation = args.pixel_seperation

    cv2.namedWindow('Contour', cv2.WINDOW_AUTOSIZE)
    if args.show_transform:
        cv2.namedWindow('Transform',cv2.WINDOW_AUTOSIZE)

    detect_count = settings["detectCount"] if settings["detectCount"] != 0 else args.detect_count
    detect_closest = settings["detectClosest"] if settings["detectClosest"] != 0 else args.detect_closest
    detect_range = settings["detectRange"] if settings["detectRange"] != [0,0] else args.detect_range

    if settings["detectionType"] != "None":
        detection_type = settings["detectionType"]
    else:
        if args.detect_largest:
            detection_type = "Largest"
        elif args.detect_closest:
            detection_type = "Closest"
        elif args.detect_range:
            detection_type = "Range"
        else:
            detection_type = None

    bound = settings["showBoundingBox"] == True or args.bounding_box == True

    if args.set_pixel_factor:
        detection_type = "Closest"
        detect_closest = 320
        bound = True

    while(True):
        # Capture frame-by-frame
        ret, frame = cam.read()
        
        if args.select_roi:
            crop_settings = cv2.selectROI(frame)
            print(crop_settings)
            cv2.destroyAllWindows()
            break

        if settings["cropImage"] == True:
            r = settings["cropDimensions"]
            frame = frame[r[1]:(r[1]+r[3]), r[0]:(r[0]+r[2])]

        cnts, img, trfm = contour(frame, settings)

        if args.show_all:
            out = img.copy()
        else:
            out = frame.copy()

        if len(cnts) != 0:
            if detection_type == "Largest":
                largest = largest_from_array(cnts,detect_count, seperation)
                if args.label:
                    out = label_contours(out,largest, args.show_size, measure)
                out,_ = measure_contours(out,largest, bound, measure)

            if detection_type == "Closest": 
                closest = closest_from_array(cnts, detect_closest, detect_count, seperation)
                if args.label:
                    out = label_contours(out,closest, args.show_size, measure)
                out,dY = measure_contours(out,closest, bound, measure)

            if detection_type == "Range":
                in_range = range_from_array(cnts, detect_range)
                largest_in_range = largest_from_array(in_range,detect_count, seperation)
                if args.label:
                    out = label_contours(out, largest_in_range, args.show_size, measure)
                out,_ = measure_contours(out,largest_in_range, bound, measure)

        h,w,_ = out.shape
        cv2.putText(out, "Res:{0}x{1}".format(w,h),
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)

        if args.detect_blob:
            out = detect_blob(out)

        if args.snap_shot:
            cv2.imwrite('out.jpg',out)
            cv2.imwrite('trfm.jpg',trfm)
            break
        else:
            dbg = cv2.resize(trfm, (0, 0), None, .30, .30)
            cv2.imshow('Contour', out)
            if args.show_transform:
                window_alignment = int(res[0])+100
                cv2.imshow('Transform',dbg)
                cv2.moveWindow('Transform',window_alignment,0)

        if args.set_pixel_factor:
            settings["pixelConversionFactor"] = 1.49/float(dY)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if args.set_pixel_factor:
        with open(settings_file, 'w') as f:
            json.dump(settings,f, indent=4)

    if args.select_roi:
        print("Saved {0} to crop settings".format(crop_settings))
        settings["cropDimensions"] = crop_settings
        settings["cropImage"] = True
        with open(settings_file, 'w') as f:
            json.dump(settings,f, indent=4)

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()

    
