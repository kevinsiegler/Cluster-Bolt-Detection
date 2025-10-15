# scripts/visualize_labels.py
import cv2, os, random
from pathlib import Path

DS = "../dataset"
img_dirs = [os.path.join(DS,"images/train"), os.path.join(DS,"images/val")]
for idir in img_dirs:
    imgs = [f for f in os.listdir(idir) if f.lower().endswith((".jpg",".png"))]
    if not imgs: continue
    sample = random.sample(imgs, min(5, len(imgs)))
    for fn in sample:
        img_path = os.path.join(idir, fn)
        lbl_path = os.path.join(DS, "labels", "train" if "train" in idir else "val", fn.rsplit(".",1)[0]+".txt")
        img = cv2.imread(img_path)
        h,w = img.shape[:2]
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    cls, xc, yc, bw, bh = line.strip().split()
                    cls = int(cls); xc=float(xc); yc=float(yc); bw=float(bw); bh=float(bh)
                    x1 = int((xc - bw/2)*w); y1 = int((yc - bh/2)*h)
                    x2 = int((xc + bw/2)*w); y2 = int((yc + bh/2)*h)
                    cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),1)
                    cv2.putText(img, str(cls), (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
        cv2.imshow("img", img)
        cv2.waitKey(0)
cv2.destroyAllWindows()
