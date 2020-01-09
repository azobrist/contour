import numpy as np

max_picam_resolution = (3240,2464)
resolutions = { "max": max_picam_resolution, 
                "high": np.multiply(max_picam_resolution,0.75), 
                "medium": np.multiply(max_picam_resolution,0.50), 
                "low": np.multiply(max_picam_resolution,0.25)}

def calc_resolution_factor(resolutionY):
    return resolutionY*3.04/2.76

#this is the pipeline used to communicate with the V2 picam module using jetson
def gstreamer_pipeline(
    capture_width=3240,
    capture_height=2464,
    display_width=3240,
    display_height=2464,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


