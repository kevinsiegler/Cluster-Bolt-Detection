import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# ============================================
# Hilfsfunktionen für Augmentationen
# ============================================

def rotate(img, bboxes, angle=5):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle,1)
    img_rot = cv2.warpAffine(img,M,(w,h))
    new_boxes = []
    for box in bboxes:
        x1,y1,x2,y2 = box
        corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        ones = np.ones((4,1))
        corners_h = np.hstack([corners,ones])
        rotated = M @ corners_h.T
        rx, ry = rotated[0], rotated[1]
        new_boxes.append([min(rx),min(ry),max(rx),max(ry)])
    return img_rot,new_boxes

def translate(img, bboxes, tx=0.1, ty=0.1):
    h,w = img.shape[:2]
    dx = tx*w
    dy = ty*h
    M = np.float32([[1,0,dx],[0,1,dy]])
    img_t = cv2.warpAffine(img,M,(w,h))
    new_boxes = [[x1+dx,y1+dy,x2+dx,y2+dy] for x1,y1,x2,y2 in bboxes]
    return img_t,new_boxes

def scale(img,bboxes,scale_factor=1.2):
    h,w = img.shape[:2]
    img_s = cv2.resize(img,(int(w*scale_factor),int(h*scale_factor)))
    new_boxes = [[x1*scale_factor,y1*scale_factor,x2*scale_factor,y2*scale_factor] for x1,y1,x2,y2 in bboxes]
    return img_s,new_boxes

def flip_lr(img,bboxes):
    img_f = cv2.flip(img,1)
    w = img.shape[1]
    new_boxes = [[w-x2,y1,w-x1,y2] for x1,y1,x2,y2 in bboxes]
    return img_f,new_boxes

def hsv(img,h=0.015,s=0.7,v=0.6):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_img[...,0] = (hsv_img[...,0]+h*180)%180
    hsv_img[...,1] = np.clip(hsv_img[...,1]*s,0,255)
    hsv_img[...,2] = np.clip(hsv_img[...,2]*v,0,255)
    return cv2.cvtColor(hsv_img.astype(np.uint8),cv2.COLOR_HSV2RGB)

def draw_bboxes(img,bboxes,color=(255,0,0),thickness=6):
    img_copy = img.copy()
    for box in bboxes:
        x1,y1,x2,y2 = map(int,box)
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),color,thickness)
    return img_copy

def copy_paste(img,bboxes,offset=20):
    img_cp = img.copy()
    h,w = img.shape[:2]
    new_boxes = []
    for box in bboxes:
        x1,y1,x2,y2 = map(int,box)
        obj = img[y1:y2,x1:x2]
        dx = min(w-x2-1,offset)
        dy = min(h-y2-1,offset)
        x_new = min(x1+dx,w-1)
        y_new = min(y1+dy,h-1)
        x_end = min(x_new + (x2-x1),w-1)
        y_end = min(y_new + (y2-y1),h-1)
        img_cp[y_new:y_end,x_new:x_end] = cv2.resize(obj,(x_end-x_new,y_end-y_new))
        new_boxes.append([x_new,y_new,x_end,y_end])
    return img_cp,bboxes+new_boxes

def mosaic(img):
    h,w = img.shape[:2]
    cut_h, cut_w = h//2, w//2
    mosaic_img = np.zeros_like(img)
    # 4 Quadranten aus zufälligen Positionen
    for i, (y,x) in enumerate([(0,0),(0,cut_w),(cut_h,0),(cut_h,cut_w)]):
        ry = random.randint(0,h-cut_h)
        rx = random.randint(0,w-cut_w)
        mosaic_img[y:y+cut_h, x:x+cut_w] = img[ry:ry+cut_h, rx:rx+cut_w]
    return mosaic_img

# ============================================
# 1️⃣ Originalbild + Label laden
# ============================================

img_path = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\images\train\67d07fb38607021bc5bfa1de.jpg"
label_path = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\labels\train\67d07fb38607021bc5bfa1de.txt"

img = cv2.imread(img_path)
h,w = img.shape[:2]

# YOLO Labels laden
bboxes = []
with open(label_path,"r") as f:
    for line in f:
        parts = line.strip().split()
        cls_id = int(parts[0])
        x_c, y_c, bw, bh = map(float, parts[1:])
        x1 = (x_c - bw/2)*w
        y1 = (y_c - bh/2)*h
        x2 = (x_c + bw/2)*w
        y2 = (y_c + bh/2)*h
        bboxes.append([x1,y1,x2,y2])

# ============================================
# 2️⃣ Alle 8 Augmentationen erzeugen
# ============================================

augmented = []
titles = []

# Original
augmented.append(draw_bboxes(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),bboxes,thickness=6))
titles.append("Original: Basisbild, unverändert")

# Rotation
rot_img, rot_boxes = rotate(img,bboxes,5)
augmented.append(draw_bboxes(cv2.cvtColor(rot_img,cv2.COLOR_BGR2RGB),rot_boxes,thickness=6))
titles.append("Rotation: Bild leicht gedreht (5°)")

# Translate
tr_img, tr_boxes = translate(img,bboxes,0.1,0.1)
augmented.append(draw_bboxes(cv2.cvtColor(tr_img,cv2.COLOR_BGR2RGB),tr_boxes,thickness=6))
titles.append("Translate: 10% nach rechts und unten verschoben")

# Scale
sc_img, sc_boxes = scale(img,bboxes,1.2)
augmented.append(draw_bboxes(cv2.cvtColor(sc_img,cv2.COLOR_BGR2RGB),sc_boxes,thickness=6))
titles.append("Scale: Bild auf 120% vergrößert")

# Flip horizontal
flip_img, flip_boxes = flip_lr(img,bboxes)
augmented.append(draw_bboxes(cv2.cvtColor(flip_img,cv2.COLOR_BGR2RGB),flip_boxes,thickness=6))
titles.append("Flip LR: Bild horizontal gespiegelt")

# HSV V
hsv_v_img = hsv(img,v=0.6)
augmented.append(draw_bboxes(hsv_v_img,bboxes,thickness=6))
titles.append("HSV V: Helligkeit reduziert auf 60%")

# HSV S
hsv_s_img = hsv(img,s=0.7)
augmented.append(draw_bboxes(hsv_s_img,bboxes,thickness=6))
titles.append("HSV S: Sättigung reduziert auf 70%")

# CopyPaste Simulation
cp_img, cp_boxes = copy_paste(img,bboxes)
augmented.append(draw_bboxes(cv2.cvtColor(cp_img,cv2.COLOR_BGR2RGB),cp_boxes,thickness=6))
titles.append("CopyPaste: Schrauben leicht dupliziert")

# Mosaic
mosaic_img = mosaic(img)
augmented.append(draw_bboxes(cv2.cvtColor(mosaic_img,cv2.COLOR_BGR2RGB),bboxes,thickness=6))
titles.append("Mosaic: 4 zufällige Bildausschnitte kombiniert")

# ============================================
# 3️⃣ Grid Plot 3x3 mit weißem Hintergrund für Titel
# ============================================
fig, axes = plt.subplots(3,3,figsize=(12,12))
axes = axes.flatten()

# Bilder ggf. skalieren
scale_factor = 0.6

for ax,img_aug,title in zip(axes,augmented,titles):
    # Bild skalieren
    img_resized = cv2.resize(img_aug, (int(img_aug.shape[1]*scale_factor), int(img_aug.shape[0]*scale_factor)))
    ax.imshow(img_resized)
    ax.set_title(title, fontsize=10, backgroundcolor='white')
    ax.axis("off")

plt.tight_layout()
plt.show()
