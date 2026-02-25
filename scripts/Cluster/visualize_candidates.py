# -*- coding: utf-8 -*-
import cv2
import os
import argparse

# Farben für die Klassen definieren (im BGR-Format)
# Klasse 0: Vorhandene Schraube -> Grün
# Klasse 1: Kandidat für fehlende Schraube -> Gelb
CLASS_COLORS = {
    0: (0, 255, 0),   # Grün
    1: (0, 255, 255), # Gelb
}
# Dicke der Bounding Box, für gute Sichtbarkeit
BOX_THICKNESS = 3

class ImageViewer:
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image
        self.h, self.w = image.shape[:2]
        
        self.zoom = 1.0
        self.center_x = self.w / 2
        self.center_y = self.h / 2
        
        self.mouse_down = False
        self.mouse_start_x = 0
        self.mouse_start_y = 0
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.mouse_start_x = x
            self.mouse_start_y = y
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_down:
                dx = x - self.mouse_start_x
                dy = y - self.mouse_start_y
                
                try:
                    rect = cv2.getWindowImageRect(self.window_name)
                    win_w, win_h = rect[2], rect[3]
                    if win_w <= 0 or win_h <= 0: win_w, win_h = 1920, 1080
                except:
                    win_w, win_h = 1920, 1080
                
                view_w = self.w / self.zoom
                view_h = self.h / self.zoom
                
                scale_x = view_w / win_w
                scale_y = view_h / win_h
                
                self.center_x -= dx * scale_x
                self.center_y -= dy * scale_y
                
                self.mouse_start_x = x
                self.mouse_start_y = y
                self.clamp_center()
                
        elif event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except AttributeError:
                delta = flags 
            
            factor = 1.1
            if delta > 0: self.zoom *= factor
            else: self.zoom /= factor
            
            if self.zoom < 1.0: self.zoom = 1.0
            self.clamp_center()

    def clamp_center(self):
        view_w = self.w / self.zoom
        view_h = self.h / self.zoom
        
        min_x, max_x = view_w / 2, self.w - view_w / 2
        min_y, max_y = view_h / 2, self.h - view_h / 2
        
        self.center_x = self.w / 2 if view_w >= self.w else max(min_x, min(self.center_x, max_x))
        self.center_y = self.h / 2 if view_h >= self.h else max(min_y, min(self.center_y, max_y))

    def show(self):
        print("Steuerung: Mausrad zum Zoomen, Linke Maustaste + Ziehen zum Verschieben.")
        print("Drücken Sie eine beliebige Taste, um das Fenster zu schließen.")
        while True:
            view_w, view_h = self.w / self.zoom, self.h / self.zoom
            x1, y1 = int(self.center_x - view_w / 2), int(self.center_y - view_h / 2)
            x2, y2 = int(self.center_x + view_w / 2), int(self.center_y + view_h / 2)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(self.w, x2), min(self.h, y2)
            
            cv2.imshow(self.window_name, self.image[y1:y2, x1:x2])
            if cv2.waitKey(10) != -1: break
        cv2.destroyAllWindows()

def zeichne_boxen_und_zeige_bild(image_pfad, label_pfad):
    """Lädt ein Bild und zeichnet Bounding Boxes aus einer YOLO-Label-Datei."""
    image = cv2.imread(image_pfad)
    if image is None:
        print(f"Fehler: Bild konnte nicht von {image_pfad} gelesen werden.")
        return

    # Wenn keine Label-Datei existiert, nur das Bild anzeigen.
    if not os.path.exists(label_pfad):
        print(f"Info: Keine Label-Datei gefunden unter {label_pfad}. Zeige nur das Originalbild an.")
    else:
        h, w, _ = image.shape
        with open(label_pfad, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    print(f"Warnung: Ungültige Zeile in Label-Datei übersprungen: {line.strip()}")
                    continue

                # Konvertiere YOLO-Format in Pixel-Koordinaten
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                # Farbe für die Klasse holen (Standard: Rot für unbekannte Klassen)
                color = CLASS_COLORS.get(class_id, (0, 0, 255)) 

                # Zeichne die Bounding Box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # Eine Legende für die Farben hinzufügen (etwas größer oben links)
    cv2.rectangle(image, (5, 5), (235, 85), (0, 0, 0), -1) # Hintergrund
    # Eintrag für Klasse 0
    cv2.rectangle(image, (15, 15), (45, 45), CLASS_COLORS[0], -1)
    cv2.putText(image, "Schraube", (55, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Eintrag für Klasse 1
    cv2.rectangle(image, (15, 50), (45, 80), CLASS_COLORS[1], -1)
    cv2.putText(image, "Fehlend", (55, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    viewer = ImageViewer(f"Bild: {os.path.basename(image_pfad)}", image)
    viewer.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visualisiert Ground-Truth-Labels auf einem Bild mit farbkodierten Bounding Boxes. "
                    "Zeigt Schrauben (Klasse 0) und fehlende Schrauben (Klasse 1) an."
    )
    parser.add_argument(
        '--image_id', 
        type=str, 
        required=False, 
        help="Die ID des anzuzeigenden Bildes (z.B. '67d0863ea3ef7e65e543fc42')."
    )
    
    args = parser.parse_args()

    # Wenn keine ID über Argumente übergeben wurde, frage den Benutzer
    if args.image_id:
        image_id = args.image_id
    else:
        image_id = input("Bitte geben Sie die Image ID ein: ").strip()

    # Pfade festlegen (wie angefordert)
    image_dir = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\images\val"
    label_dir = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\labels\val"

    # Dateipfade erstellen (sucht nach gängigen Bild-Erweiterungen)
    image_pfad = None
    for ext in ['.jpg', '.jpeg', '.png']:
        potenzieller_pfad = os.path.join(image_dir, image_id + ext)
        if os.path.exists(potenzieller_pfad):
            image_pfad = potenzieller_pfad
            break
    
    if image_pfad is None:
        print(f"Fehler: Bild mit ID '{image_id}' wurde im Ordner '{image_dir}' nicht gefunden.")
        return

    label_pfad = os.path.join(label_dir, image_id + '.txt')

    zeichne_boxen_und_zeige_bild(image_pfad, label_pfad)

if __name__ == '__main__':
    main()