import cv2, os

DS = "../dataset"
IMG_DIRS = [os.path.join(DS, "images/train"), os.path.join(DS, "images/val")]

MAX_SIZE = 1000  # maximale Breite oder Höhe für die Anzeige

def visualize_image_by_id(image_id):
    img_path = None
    lbl_path = None
    split_found = None

    # Suche das Bild im train- oder val-Ordner
    for idir in IMG_DIRS:
        for ext in [".jpg", ".png"]:
            potential_path = os.path.join(idir, image_id + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                split_found = "train" if "train" in idir else "val"
                lbl_path = os.path.join(DS, "labels", split_found, image_id + ".txt")
                break
        if img_path:
            break

    if not img_path:
        print(f"Bild mit ID {image_id} nicht gefunden in train oder val.")
        return

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Berechne Skalierungsfaktor
    scale = min(MAX_SIZE / w, MAX_SIZE / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = line.strip().split()
                cls = int(cls)
                xc, yc, bw, bh = map(float, (xc, yc, bw, bh))

                # Bounding Box auf die neue Bildgröße skalieren
                x1 = int((xc - bw/2) * new_w)
                y1 = int((yc - bh/2) * new_h)
                x2 = int((xc + bw/2) * new_w)
                y2 = int((yc + bh/2) * new_h)

                # Linienbreite und Schriftgröße proportional zur Bildgröße
                thickness = max(1, int(new_w / 500))
                font_scale = max(0.4, new_w / 1000 * 0.4)

                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0,255,0), thickness)
                cv2.putText(img_resized, str(cls), (x1, max(10, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)

    cv2.imshow("img", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_image_by_id("6764b34ed9f572ca96e519fb")
