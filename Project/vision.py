import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxHeight = int(max(heightA, heightB))

    margin = 20
    
    dst = np.array([
        [margin, margin],
        [margin + maxWidth - 1, margin],
        [margin + maxWidth - 1, margin + maxHeight - 1],
        [margin, margin + maxHeight-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth + 2*margin, maxHeight + 2*margin))
    return warped
    
def detect_walls(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 31)
    
    analysis = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    
    walls = np.zeros(thresh.shape, dtype="uint8")
    min_area = 6000
    
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]  

        if (area > min_area) :
            componentMask = (label_ids == i).astype("uint8") * 255
            walls = cv2.bitwise_or(walls, componentMask)
            
    return walls
    
def to_grid(walls, coordinates, rows=10, cols=10):
    h, w = walls.shape
    grid = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            y_start = int(r*h/rows)
            y_end   = int((r + 1)*h/rows)
            x_start = int(c*w/cols)
            x_end   = int((c + 1)*w/cols)
            
            cell = walls[y_start:y_end, x_start:x_end]
            if np.mean(cell) > 100:
                grid[r, c] = 1 
            else:
                grid[r, c] = 0
    
    if coordinates is not None:
        cell_h = h // rows
        cell_w = w // cols
        if cell_h != 0 and cell_w != 0:
            center, _ = coordinates
            r = min(center[1] // cell_h, rows - 1)
            c = min(center[0] // cell_w, cols - 1)
            grid[r, c] = 2

    return grid

def draw_grid(grid, cell_px=40):
    img = np.where(grid == 1, 0, 255).astype(np.uint8)
    img[grid == 2] = 100
    
    return img
    
def find_light(mask, min_area=20):
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contour:
        return None

    largest = max(contour, key=cv2.contourArea)
    
    if cv2.contourArea(largest) < min_area:
        return None
    
    M = cv2.moments(largest)

    if M['m00'] == 0:
        return None
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    return cx, cy
    
def detect_thymio(img):
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_g = np.array([35, 80, 80])
    upper_g = np.array([85, 255, 255])
    lower_b = np.array([90, 80, 80])
    upper_b = np.array([120, 255, 255])
    
    mask_g = cv2.inRange(hsv, lower_g, upper_g)
    mask_b = cv2.inRange(hsv, lower_b, upper_b)
    
    kernel = np.ones((5, 5), np.uint8)
    
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kernel)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel)

    light_g = find_light(mask_g)
    light_b = find_light(mask_b)
    
    if light_g == None or light_b == None:
        return None

    cx_g, cy_g = light_g
    cx_b, cy_b = light_b

    center_g = [cx_g, cy_g]
    center_b = [cx_b, cy_b]
    
    theta = np.arctan2(cy_g - cy_b, cx_g - cx_b)
    center = [(cx_g + cx_b)//2, (cy_g + cy_b)//2]
    
    return center, theta

def draw_direction(img, coordinates):
    if coordinates == None :
        return img

    center, theta = coordinates
    
    img = cv2.circle(img, center, 5, (0, 0, 255), -1)

    x_end = int(center[0] + 100*np.cos(theta))
    y_end = int(center[1] + 100*np.sin(theta))

    img = cv2.arrowedLine(img, center, (x_end, y_end), (100, 100, 0), 5)

    return img