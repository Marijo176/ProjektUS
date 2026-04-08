import cv2
import urllib.request
import numpy as np
import math
 
# --- KONFIGURACIJA ---
url = 'http://192.168.222.73/capture'
MIN_AREA_THRESHOLD = 800
# ---------------------
 
 
def classify_shape(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
 
    if area < MIN_AREA_THRESHOLD or perimeter == 0:
        return None, None
 
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
 
    circularity = (4 * math.pi * area) / (perimeter * perimeter)
 
    # 🔵 KRUG
    if circularity > 0.82 and len(approx) > 6:
        return "KRUG", circularity
 
    # 🔺 TROKUT
    if len(approx) == 3:
        return "TROKUT", circularity
 
    # 🟩 KVADRAT
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
 
        if h == 0:
            return "NEPOZNATO", circularity
 
        aspect_ratio = float(w) / h
        rect_area = w * h
        fill_ratio = area / rect_area if rect_area > 0 else 0
 
        if 0.90 <= aspect_ratio <= 1.10 and fill_ratio > 0.75:
            return "KVADRAT", circularity
 
    return "NEPOZNATO", circularity
 
 
while True:
    try:
        img_resp = urllib.request.urlopen(url, timeout=5)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
 
        if frame is None:
            continue
 
        frame_display = frame.copy()
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
 
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        for cnt in contours:
            shape_class, circ_val = classify_shape(cnt)
 
            if shape_class is None:
                continue
 
            if shape_class in ["KVADRAT", "KRUG", "TROKUT"]:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
 
                    # boje po obliku
                    if shape_class == "KVADRAT":
                        color = (0, 255, 0)      # zeleno
                    elif shape_class == "TROKUT":
                        color = (0, 165, 255)    # narančasto
                    else:
                        color = (255, 0, 0)      # plavo
 
                    cv2.drawContours(frame_display, [cnt], -1, color, 2)
                    cv2.putText(frame_display, shape_class,
                                (cx - 40, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 
        cv2.imshow("ESP32-CAM Analiza", frame_display)
        cv2.imshow("Rubovi", edges)
 
    except Exception as e:
        print(f"Greska kod dohvacanja: {e}")
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cv2.destroyAllWindows()