HUE_MIN = 20
HUE_MAX = 160
SAT_MIN = 100
SAT_MAX = 255
VAL_MIN = 200
VAL_MAX = 256
        
channels = {
    'hue': None,
    'saturation': None,
    'value': None,
    'laser': None,
}
def threshold_image(channel):
    if channel == "hue":
        minimum = HUE_MIN 
        maximum = HUE_MAX
    elif channel == "saturation":
        minimum = SAT_MIN
        maximum = SAT_MAX
    elif channel == "value":
        minimum = VAL_MIN
        maximum = VAL_MAX
   (t, tmp) = cv2.threshold(
        channels[channel], # src
        maximum, # threshold value
        0, # we dont care because of the selected type
        cv2.THRESH_TOZERO_INV #t type
    )

    (t, channels[channel]) = cv2.threshold(
        tmp, # src
        minimum, # threshold value
        255, # maxvalue
        cv2.THRESH_BINARY # type
    )

    if channel == 'hue':
        # only works for filtering red color because the range for the hue is split
        channels['hue'] = cv2.bitwise_not(channels['hue'])

def detect(frame):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # split the video frame into color channels
    h, s, v = cv2.split(hsv_img)
    channels['hue'] = h
    channels['saturation'] = s
    channels['value'] = v

    # Threshold ranges of HSV components; storing the results in place
    threshold_image("hue")
    threshold_image("saturation")
    threshold_image("value")

    # Perform an AND on HSV components to identify the laser!
    channels['laser'] = cv2.bitwise_and(channels['hue'], channels['value'])
  
    channels['laser'] = cv2.bitwise_and(channels['saturation'], channels['laser'])
    
    #track
    mask = channels['laser'] 
    center = None
    countours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # only proceed if at least one contour was found
    if len(countours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(countours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        moments = cv2.moments(c)
        if moments["m00"] > 0:
            center = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        else:
            center = int(x), int(y)
    	
