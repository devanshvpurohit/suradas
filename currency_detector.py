import cv2
import numpy as np
import re
from collections import deque

class SmoothIndianCurrencyDetector:
    """
    Real-time Indian Banknote Detector & Classifier.
    Combines:
    1. Banknote contour & bounding box localization
    2. Numerical & RBI script OCR extraction
    3. Multi-band HSV color profile verification
    4. Temporal voting filter for flicker-free recognition
    """
    def __init__(self, ocr_reader):
        self.ocr = ocr_reader
        self.history = deque(maxlen=6)
        
        # RBI Mahatma Gandhi New Series Denomination Profiles
        self.profiles = {
            "500": {"name": "500 Rupees", "hue_range": (35, 85),  "color_name": "Stone Grey / Green"},
            "200": {"name": "200 Rupees", "hue_range": (10, 25),  "color_name": "Bright Orange-Yellow"},
            "100": {"name": "100 Rupees", "hue_range": (130, 165), "color_name": "Lavender / Violet"},
            "50":  {"name": "50 Rupees",  "hue_range": (90, 125), "color_name": "Fluorescent Blue"},
            "20":  {"name": "20 Rupees",  "hue_range": (26, 36),  "color_name": "Greenish-Yellow"},
            "10":  {"name": "10 Rupees",  "hue_range": (0, 10),   "color_name": "Chocolate Brown"}
        }

    def extract_banknote_roi(self, frame):
        """Finds the banknote in the frame using contour analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        min_area = (w * h) * 0.08  # Note must occupy at least 8% of frame

        best_box = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area and area > max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                # Indian banknotes have aspect ratios between 1.3 and 2.5
                if 1.2 <= aspect_ratio <= 2.8:
                    max_area = area
                    best_box = (x, y, bw, bh)

        if best_box is not None:
            x, y, bw, bh = best_box
            crop = frame[y:y+bh, x:x+bw]
            return crop, best_box
        return frame, (0, 0, w, h)

    def analyze_color(self, crop):
        """Analyzes color hue of the banknote."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Mask out white paper / black backgrounds
        valid_pixels = (s > 40) & (v > 50) & (v < 235)
        if np.sum(valid_pixels) < 500:
            return None

        avg_hue = np.mean(h[valid_pixels])
        
        for denom, data in self.profiles.items():
            low, high = data["hue_range"]
            if low <= avg_hue <= high:
                return denom
        return None

    def read_ocr_denomination(self, crop):
        """Reads numerical denomination printed on the note."""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        texts = self.ocr.readtext(rgb, detail=0)
        
        combined_text = " ".join(texts).upper()
        
        # Check explicit numerals
        matches = re.findall(r'\b(500|200|100|50|20|10)\b', combined_text)
        if matches:
            return matches[0]
            
        # Check Hindi / English words if numeral wasn't clean
        if "FIVE HUNDRED" in combined_text or "5OO" in combined_text:
            return "500"
        elif "TWO HUNDRED" in combined_text or "2OO" in combined_text:
            return "200"
        elif "ONE HUNDRED" in combined_text or "1OO" in combined_text:
            return "100"
        elif "FIFTY" in combined_text:
            return "50"
        elif "TWENTY" in combined_text:
            return "20"
        elif "TEN" in combined_text:
            return "10"

        return None

    def detect(self, frame):
        """
        Executes full pipeline:
        Returns: (detected_denomination_str, bounding_box, confidence_score)
        """
        crop, bbox = self.extract_banknote_roi(frame)

        # Step 1: High-accuracy OCR read
        ocr_result = self.read_ocr_denomination(crop)
        
        # Step 2: Color profile analysis
        color_result = self.analyze_color(crop)

        final_val = None
        conf = 0.5

        if ocr_result and color_result:
            if ocr_result == color_result:
                final_val = ocr_result
                conf = 0.95  # Dual verified
            else:
                final_val = ocr_result
                conf = 0.80
        elif ocr_result:
            final_val = ocr_result
            conf = 0.85
        elif color_result:
            final_val = color_result
            conf = 0.70

        if final_val:
            self.history.append(final_val)
        else:
            self.history.append(None)

        # Smooth decision: Most common result in rolling buffer
        valid_votes = [v for v in self.history if v is not None]
        if len(valid_votes) >= 2:
            from collections import Counter
            most_common, count = Counter(valid_votes).most_common(1)[0]
            if count >= 2:
                name = self.profiles.get(most_common, {}).get("name", f"{most_common} Rupees")
                return name, bbox, conf

        return None, bbox, 0.0
