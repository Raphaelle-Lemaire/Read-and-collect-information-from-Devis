#pip install opencv-python numpy pillow pytesseract ultralytics torch openpyxl pdf2image

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from ultralytics import YOLO
import torch
from pathlib import Path
import json
import re
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd
from pdf2image import convert_from_path
import time


class OCR:  
    def __init__(self, model_type='yolov8', kernel_selection=15):
        self.model_type = model_type
        self.model = None
        self.template_classes = ['two_col', 'three_col_anp', 'three_col_nap']
        self.kernel_selection = kernel_selection
        
        self.payment_keywords = [
            'total', 'sub-total', 'subtotal', 'sous-total',
            'tva', 'vat', 'taxe', 'tax',
            'service', 'pourboire', 'tip',
            'remise', 'discount', 'réduction',
            'à payer', 'montant', 'amount'
        ]
        print(f"Système initialisé avec modèle: {model_type}")
    
    def load_model(self, model_path=None):
        if self.model_type == 'yolov8':
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                self.model = YOLO('yolov8n.pt')
        else:
            raise NotImplementedError(f"Modèle {self.model_type} non implémenté")
    
    # ========== ÉTAPE 1: PRÉTRAITEMENT ==========

    def comparer_pages(self, img1, img2):
        if len(img1.shape) == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        h, w = img1.shape[:2]
        
        y1, y2 = int(h * 0.1), int(h * 0.35)
        x1, x2 = int(w * 0.45), int(w * 0.95)
        
        cadre_p1 = img1[y1:y2, x1:x2]

        mask_roi = np.zeros_like(img1)
        mask_roi[y1:y2, x1:x2] = 255
        
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img1, mask=mask_roi)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 5:
            return "AUTRE"

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 10:
            return "AUTRE"

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is not None:
            obj_corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
            try:
                dst_corners = cv2.perspectiveTransform(obj_corners, M)
                x_min, y_min = np.int32(dst_corners.min(axis=0).ravel())
                x_max, y_max = np.int32(dst_corners.max(axis=0).ravel())
                
                y_min, y_max = max(0, y_min), min(h, y_max)
                x_min, x_max = max(0, x_min), min(w, x_max)
                
                cadre_p2_detecte = img2[y_min:y_max, x_min:x_max]
            except:
                pass 

            nb_points_coherents = sum(mask.flatten().tolist())
            score_coherence = nb_points_coherents / len(matches)
        else:
            return "AUTRE"

        print(f"Points retrouvés : {nb_points_coherents} | Score : {score_coherence:.2f}")
        if nb_points_coherents > 10 and score_coherence > 0.05:
            return "SUITE"
        else:
            kp1, des1 = orb.detectAndCompute(img1, None)
            matches_mask = mask.flatten().tolist() 
            nb_points_coherents = sum(matches_mask)
            matches = bf.match(des1, des2)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.flatten().tolist()
            nb_points_coherents = sum(matches_mask)

            score_coherence = nb_points_coherents / len(matches) if len(matches) > 0 else 0 
            print(f"Points retrouvés v2: {nb_points_coherents} | Score v2: {score_coherence:.2f}")
            if nb_points_coherents > 10 and score_coherence > 0.05: 
                return "NOUVEAU"
        return "AUTRE"

    def charge_image(self, image_path, type=type):
        image_path = Path(image_path)
        nb_pages = None
        dicoPages = []
        if not image_path.exists():
            raise FileNotFoundError(f"Le fichier n'existe pas: {image_path.absolute()}")
        if type == "png":
            img = cv2.imread(str(image_path.absolute()))
            
            if img is not None:
                try:
                    pil_img = Image.open(image_path)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    dicoPages.append({"page": img, "type": type})
                except Exception as e:
                    raise ValueError(f"Impossible de charger l'image: {image_path}\nErreur: {e}")
            
        elif type == "pdf":
            pages = convert_from_path(str(image_path.absolute()), 300)
            if not pages:
                raise ValueError("Le PDF est vide ou illisible")
            pil_img = pages[0].convert('RGB') 
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            nb_pages = len(pages)
            dicoPages.append({"page": img, "type": "NOUVEAU"})

            if nb_pages>1:
                for page in pages[1:]:
                    pil_img2 = page.convert('RGB') 
                    img2 = cv2.cvtColor(np.array(pil_img2), cv2.COLOR_RGB2BGR)
                    type = self.comparer_pages(img, img2)
                    dicoPages.append({"page": img2, "type": type})
        return dicoPages

    
    def preprocess_image(self, img):
        original_img = img.copy()
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(gray, [c], -1, (255, 255, 255), 5)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(gray, [c], -1, (255, 255, 255), 5)
        edges = cv2.Canny(gray, 50, 150)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        dilated1 = cv2.dilate(edges, kernel1, iterations=1)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_selection, 1))
        dilated2 = cv2.dilate(dilated1, kernel2, iterations=1)

        contours, _ = cv2.findContours(dilated2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        min_width, max_width = 30, w * 0.95
        min_height, max_height = 10, h * 0.15
        min_area = 200
        
        for cnt in contours:
            x, y, width, height = cv2.boundingRect(cnt)
            area = width * height

            if (area > min_area and 
                min_width < width < max_width and 
                min_height < height < max_height):
                
                bounding_boxes.append({
                    'x': x, 'y': y, 
                    'width': width, 'height': height,
                    'area': area
                })

        bounding_boxes = self._handle_outliers(bounding_boxes, edges, h)
        bounding_boxes.sort(key=lambda b: (b['y'], b['x']))
        binary_template = np.zeros((h, w), dtype=np.uint8)
        for box in bounding_boxes:
            cv2.rectangle(binary_template, 
                         (box['x'], box['y']),
                         (box['x'] + box['width'], box['y'] + box['height']),
                         255, -1)
        
        return binary_template, bounding_boxes, original_img
    
    def _handle_outliers(self, bounding_boxes, edges, img_height):
        if not bounding_boxes:
            return bounding_boxes
        
        heights = [b['height'] for b in bounding_boxes]
        median_height = np.median(heights)
        threshold_height = median_height * 1.8
        
        new_boxes = []
        for box in bounding_boxes:
            if box['height'] > threshold_height:
                x, y, w, h = box['x'], box['y'], box['width'], box['height']
                region = edges[y:y+h, x:x+w]
                histogram = np.sum(region, axis=1)
                q1 = np.percentile(histogram[histogram > 0], 25) if np.any(histogram > 0) else 0
                split_points = []
                for i, val in enumerate(histogram):
                    if val < q1:
                        split_points.append(i)
                
                if split_points:
                    mid_point = split_points[len(split_points)//2]
                    
                    new_boxes.append({
                        'x': x, 'y': y,
                        'width': w, 'height': mid_point,
                        'area': w * mid_point
                    })
                    
                    new_boxes.append({
                        'x': x, 'y': y + mid_point,
                        'width': w, 'height': h - mid_point,
                        'area': w * (h - mid_point)
                    })
                else:
                    new_boxes.append(box)
            else:
                new_boxes.append(box)
        
        return new_boxes
    
    # ========== ÉTAPE 2: DÉTECTION DU TEMPLATE ==========
    
    def detect_template(self, binary_template, bounding_boxes):
        if self.model is not None:
            return self._detect_with_yolo(binary_template, bounding_boxes)

        return self._detect_with_rules(bounding_boxes)
    
    def _detect_with_yolo(self, binary_template, bounding_boxes):
        template_rgb = cv2.cvtColor(binary_template, cv2.COLOR_GRAY2RGB)
        results = self.model(template_rgb, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            best_idx = np.argmax(confidences)
            template_class = self.template_classes[int(classes[best_idx])]
            confidence = confidences[best_idx]
        else:
            print("Aucune détection YOLO - utilisation des règles")
            template_class, _ = self._detect_with_rules(bounding_boxes)
        
        return template_class, bounding_boxes
    
    def _detect_with_rules(self, bounding_boxes):
        if not bounding_boxes:
            return 'two_col', bounding_boxes
        centers = [b['x'] + b['width']//2 for b in bounding_boxes]
        centers_sorted = sorted(centers)
        clusters = []
        current_cluster = [centers_sorted[0]]
        threshold = 50
        
        for center in centers_sorted[1:]:
            if center - current_cluster[-1] < threshold:
                current_cluster.append(center)
            else:
                clusters.append(current_cluster)
                current_cluster = [center]
        clusters.append(current_cluster)
        
        num_columns = len(clusters)
        
        if num_columns == 2:
            template_class = 'two_col'
        elif num_columns >= 3:
            avg_widths = [np.mean([bounding_boxes[i]['width'] 
                         for i, c in enumerate(centers) 
                         if min(cluster) <= c <= max(cluster)])
                         for cluster in clusters[:3]]
            if avg_widths[0] < avg_widths[1] * 0.5:
                template_class = 'three_col_anp'
            else:
                template_class = 'three_col_nap'
        else:
            template_class = 'two_col'
        return template_class, bounding_boxes
    
    # ========== ÉTAPE 3: OCR ==========
    
    def perform_ocr(self, original_img, bounding_boxes):
        ocr_results = []
        
        for i, box in enumerate(bounding_boxes):
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            roi = original_img[y:y+h, x:x+w]
            if roi is None or roi.size == 0:
                print("Warning: Received an empty ROI for preprocessing. Skipping...")
            else:
                roi_processed = self._preprocess_for_ocr(roi)
                custom_config = r'--oem 3 --psm 7 -l fra+eng'
                text = pytesseract.image_to_string(roi_processed, config=custom_config)
                text = text.strip()

                if text:
                    ocr_results.append({
                        'box': box,
                        'text': text,
                        'position': (x, y, w, h)
                    })

                    # Affichage progressif
                    #if (i + 1) % 10 == 0 or i == len(bounding_boxes) - 1:
                    #    print(f"  Progression: {i+1}/{len(bounding_boxes)} régions traitées")
        
        return ocr_results
    
    def _preprocess_for_ocr(self, roi):
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        denoised = cv2.fastNlMeansDenoising(normalized, None, 10, 7, 21)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        return eroded
    
    # ========== ÉTAPE 4: EXTRACTION D'INFORMATIONS ==========
    
    def extract_information(self, ocr_results, template_class):
        menu_items = []
        payment_info = {}
        payment_section = False
        
        for result in ocr_results:
            text = result['text']
            if self._is_payment_keyword(text):
                payment_section = True
            
            if payment_section:
                self._extract_payment_info(text, payment_info)
            else:
                item = self._extract_menu_item(text, template_class)
                if item and item.get('price', 0) > 0:
                    menu_items.append(item)
        return menu_items, payment_info
    
    def _is_payment_keyword(self, text):
        text_lower = text.lower()
        
        for keyword in self.payment_keywords:
            ratio = SequenceMatcher(None, keyword, text_lower).ratio()
            if ratio > 0.8:
                return True
        
        return False
    
    def _extract_menu_item(self, text, template_class):
        words = text.split()
        
        if len(words) < 2:
            return None
        
        price = self._extract_price(words[-1])
        if price is None:
            return None
        
        item = {'price': price}
        
        if template_class == 'two_col':
            item['name'] = ' '.join(words[:-1])
            item['quantity'] = 1
            
        elif template_class == 'three_col_anp':
            if len(words) >= 3:
                qty = self._extract_quantity(words[0])
                item['quantity'] = qty if qty else 1
                item['name'] = ' '.join(words[1:-1])
            else:
                item['name'] = ' '.join(words[:-1])
                item['quantity'] = 1
                
        elif template_class == 'three_col_nap':
            if len(words) >= 3:
                qty = self._extract_quantity(words[-2])
                item['quantity'] = qty if qty else 1
                item['name'] = ' '.join(words[:-2])
            else:
                item['name'] = ' '.join(words[:-1])
                item['quantity'] = 1
        
        return item
    
    def _extract_price(self, text):
        patterns = [
            r'(\d+[,\.]\d{2})', 
            r'(\d+)', 
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                price_str = match.group(1).replace(',', '.')
                try:
                    return float(price_str)
                except:
                    continue
        
        return None
    
    def _extract_quantity(self, text):
        match = re.search(r'(\d+)', text)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
        return None
    
    def _extract_payment_info(self, text, payment_info):
        text_lower = text.lower()
        price = self._extract_price(text)
        if price is None:
            return

        if any(kw in text_lower for kw in ['total', 'montant']):
            payment_info['total'] = price
        elif any(kw in text_lower for kw in ['sub', 'sous']):
            payment_info['subtotal'] = price
        elif any(kw in text_lower for kw in ['tva', 'vat', 'tax']):
            payment_info['tax'] = price
        elif any(kw in text_lower for kw in ['service', 'pourboire']):
            payment_info['service'] = price
        elif any(kw in text_lower for kw in ['remise', 'discount', 'réduction']):
            payment_info['discount'] = price
    
    # ========== TRAITEMENT COMPLET ==========
    
    def process_receipt(self, image, output_dir='output', output_file='file'):
        output_path = Path(output_dir)
        output_path_json = Path(output_dir+ "/" + output_file + '.json')
        output_path.mkdir(exist_ok=True)
        binary_template, bounding_boxes, original_img = self.preprocess_image(image)
        template_path = output_path / "binary.png"
        cv2.imwrite(str(template_path), binary_template)
        template_class, template_boxes = self.detect_template(binary_template, bounding_boxes)
        ocr_results = self.perform_ocr(original_img, template_boxes)
        self._save_results(ocr_results, original_img, output_path_json)
        
        return ocr_results
        
    def _save_results(self, results, original_img, output_path):
        json_path = output_path
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON: {json_path}")

class ExtractFromOCR:
    def __init__(self):
        self.tailleHaie = "Taille de haies"
        self.tailleRonce = "Taille de ronces"
        self.tontePelouse = "Entretien Pelouse"
        self.entretienAnnuel = "Entretien Annuel"
        self.elagageArbre = "Élagage Arbres"
        self.engraisSelectif = "Engrais, Sélectif"
        self.bechage = "Béchage"
        self.bassinOrage = "Entretien Bassin d'orage"
        self.entretienMassif = "Entretien Massifs"
        self.entretienArbuste = "Entretien Arbuste"
        self.desherbage = "Désherbage"
        self.debroussaillage = "Débroussaillage"
        self.deplacement = "Déplacement Transport"
        self.transport = "Enlèvement Transport"
        self.dechets = "Traçabilité Déchets"
        self.passage = "Passage Supplémentaire"
        self.copeaux = "Fourniture de copeaux"

    def normaliser_prestation(self, texte):
        t = texte.lower()
        renvoie = ""
        if "entretien" in t and "annuel" in t:
            renvoie +=  self.entretienAnnuel + "/"
        if "haie" in t or "hate" in t or "haic" in t or "häie" in t:
            renvoie +=  self.tailleHaie + "/"
        if "ronce" in t:
            renvoie +=  self.tailleRonce + "/"
        if "tonte" in t or "pelouse" in t or 'pélouse' in t:
            renvoie += self.tontePelouse + "/"
        if "engrai" in t or "sélectif" in t:
            renvoie += self.engraisSelectif + "/"
        if "élagage" in t or "arbre" in t:
            renvoie += self.elagageArbre + "/"
        if "massif" in t:
            renvoie += self.entretienMassif + "/"
        if "copeaux" in t:
            renvoie += self.copeaux + "/"
        if "arbuste" in t:
            renvoie += self.entretienArbuste + "/"
        if "bassin" in t:
            renvoie += self.bassinOrage + "/"
        if "désherbage" in t:
            renvoie += self.desherbage + "/"
        if "débroussaillage" in t or "débroussarllage" in t:
            renvoie += self.debroussaillage + "/"
        if "béchage" in t:
            renvoie += self.bechage + "/"
        if "transport" in t and not "déplacement" in t:
            renvoie += self.transport + "/"
        if "déplacement" in t:
            renvoie += self.deplacement + "/"
        if "passages supplémentaire" in t:
            renvoie += self.passage + "/"
        if "traçabilité" in t or "déchets" in t:
            renvoie += self.dechets + "/"

        if renvoie!="":
            return renvoie[:-1]
        return texte

    def extract_quote_data(self, data, filename):
        resultat = {
            "client": "",
            "code_affaire": "",
            "num_devis": "",
            "interventions": [],
            "total_ht": 0.0,
            "tva_taux": 20.0,
            "total_ttc": 0.0
        }
        listeValeursIdentiques = []
        match_code_file = re.search(r'(EVA\d+)', filename)
        if match_code_file:
            resultat["code_affaire"] = match_code_file.group(1)
        sorted_data = sorted(data, key=lambda k: k['box']['y'])
        for item in sorted_data:
            txt = item.get('text', '').strip()
            x, y = item['box']['x'], item['box']['y']

            if resultat["num_devis"] == "":
                match_devis = re.search(r'(DE\d+|DH\d+|EVA\d+)', txt)
                if match_devis:
                    resultat["num_devis"] = match_devis.group(1)

            if x > 1100 and y < 1100 and resultat["client"] == "":
                if txt.startswith(('Mr', 'Mme', 'M.', 'ECOLE', 'ASSOCIATION', 'MAIRIE')):
                    resultat["client"] = txt
                    
            if resultat["client"] == "":
                for item in sorted_data:
                    txt = item.get('text', '').strip()
                    x, y = item['box']['x'], item['box']['y']
                    if 1100 < x < 2400 and 480 < y < 1050:
                        is_date = re.search(r'(\d{2}/\d{2}/\d{2,4})|(\d{4})', txt)
                        is_meta = any(ex in txt.upper() for ex in ["SIRET", "TVA", "TEL", "EMAIL", "ADAPTÉE", "PAGE", "DEVIS"])

                        if not is_date and not is_meta and len(txt) > 2:
                            resultat["client"] = txt
                            break
        lignes_temp = []

        for item in sorted_data:
            txt = item.get('text', '').strip()
            y_pos = item['box']['y']
            x_pos = item['box']['x']

            # Zone du tableau des prix
            if 1000 < y_pos < 3000 and x_pos < 1100:
                if any(key in txt.lower() for key in ["total", "validation", "assurance", "siret", "reporter"]):
                    continue

                m2_match = re.search(r'(\d+[\.,]?\d*)\s*(m²|m2|m\?|m\*|tm\?|m\°)', txt + 'm2', re.IGNORECASE)
                ml_match = re.search(r'(\d+[\.,]?\d*)\s*(ml|mi|m\]|m\[)', txt + 'ml', re.IGNORECASE)
                prix_total = 0.0
                prix_uni = 0.0
                nb_prest = 1

                for p_item in data:
                    if abs(p_item['box']['y'] - y_pos) < 35:
                        val_txt = p_item['text'].replace(',', '.').replace(' ', '').replace('€', '')
                        try:
                            if not re.search(r'\d', val_txt): continue
                            valeur = float(val_txt)
                            if valeur.is_integer() and 0 < valeur < 50 and p_item['box']['x'] < 1800:
                                nb_prest = int(valeur)
                            elif p_item['box']['x'] > 2000:
                                prix_total = valeur
                            elif 1400 < p_item['box']['x'] <= 2000:
                                prix_uni = valeur
                        except ValueError:
                            continue

                if (m2_match or ml_match) and prix_total == 0 and len(lignes_temp) > 0:
                    last = lignes_temp[-1]
                    if m2_match: last["surface"] = m2_match.group(1)
                    if ml_match: last["lineaire"] = ml_match.group(1)
                elif prix_total > 0:
                    if prix_uni == 0: prix_uni = prix_total / nb_prest
                    if prix_uni != prix_total and nb_prest==1: prix_uni = prix_total
                    if [prix_uni, prix_total] in listeValeursIdentiques: 
                        if not lignes_temp[listeValeursIdentiques.index([prix_uni, prix_total])]["surface"] and m2_match:
                            lignes_temp[listeValeursIdentiques.index([prix_uni, prix_total])]["surface"]= m2_match.group(1)
                        if not lignes_temp[listeValeursIdentiques.index([prix_uni, prix_total])]["lineaire"] and ml_match:
                            lignes_temp[listeValeursIdentiques.index([prix_uni, prix_total])]["lineaire"]= ml_match.group(1)
                        continue
                    
                    lignes_temp.append({
                        "nom": self.normaliser_prestation(txt),
                        "nom_original": txt,
                        "nb_prestation": nb_prest,
                        "surface": m2_match.group(1) if m2_match else None,
                        "lineaire": ml_match.group(1) if ml_match else None,
                        "prix_unitaire": round(prix_uni, 2),
                        "prix_ht": round(prix_total, 2)
                    })
                    listeValeursIdentiques.append([prix_uni, prix_total])

        resultat["interventions"] = lignes_temp
        resultat["total_ht"] = round(sum(i['prix_ht'] for i in resultat["interventions"]), 2)
        resultat["total_ttc"] = round(resultat["total_ht"] * (1 + resultat["tva_taux"]/100), 2)

        return resultat

if __name__ == "__main__":
    kernel = 25
    type = "png"
    system = OCR(model_type='yolov8', kernel_selection=kernel)
    dossier = 'Devis annuels 2024 acceptés/pdf/'
    filenames = [f.name for f in Path(dossier).glob('*.'+type)]
    toutes_les_prestations = []
    numero = 0

    i=0

    for file in filenames[i:]:
        numero+=1
        print(numero, '/', len(filenames))
        image_path = dossier + file

        codeAffaire = None
        nomClient = None

        try:
            dicoPages = system.charge_image(image_path, type=type)
            for dicoPage in dicoPages:
                if dicoPage['type'] == "AUTRE":
                    continue
                image = dicoPage['page']
                results = system.process_receipt(image, output_dir='output_results', output_file = file)
                extract = ExtractFromOCR()
                result = extract.extract_quote_data(results, file)

                if nomClient is None or codeAffaire is None or dicoPage['type'] == "NOUVEAU":
                    codeAffaire = result['code_affaire']
                    nomClient = result['client']

                for r in result['interventions']:
                    if extract.transport in r['nom'] or extract.passage in r['nom'] or extract.deplacement in r['nom']:
                        r['surface'] = None
                        r['lineaire'] = None
                    if "20" in r['nom']  or "%" in r['nom'] or "TVA" in r['nom'] or "€" in r['nom'] or "environnement" in r['nom'] or "A régler" in r['nom'] or "Un passage" in r['nom'] or "Report" in r['nom'] or r['nom'].strip().isdigit() or len(r['nom'])<=2:
                        continue
                    ligne = {
                        "Nom fichier": image_path,
                        "Code Affaire": codeAffaire,
                        "Nom Client": nomClient,
                        "Prestation": r['nom'],
                        "Prestation nom complet": r['nom_original'],
                        "Nombre d'unitée": r['nb_prestation'],
                        "Surface (m2 ou ml)": r['surface'] + 'm2 /' + r['lineaire'] + 'ml' if r['surface'] and r['lineaire'] else  r['surface'] + 'm2' if r['surface'] else r['lineaire'] + "ml" if r['lineaire'] else "",
                        "Prix Unité": r['prix_unitaire'],
                        "Prix préstation": r['prix_ht']
                    }
                    toutes_les_prestations.append(ligne)
                    
                print(f"--- Fichier : {file} ---")
                print(f"Affaire : {result['code_affaire']} | Devis : {result['num_devis']}")
                print(f"Client  : {result['client']}")
                for p in result['interventions']:
                    mesure = f" [{p['surface']} m²]" if p['surface'] else ""
                    mesure += f" [{p['lineaire']} ml]" if p['lineaire'] else ""
                    print(f"  - {p['nom']}{mesure} (Qté: {p['nb_prestation']})")
                    print(f"    PU: {p['prix_unitaire']}€ | Total HT: {p['prix_ht']}€")

        except FileNotFoundError as e:
            print(f"Fichier introuvable: {e}")
            print("Vérifiez que le chemin est correct et que le fichier existe.")
            ligne = {
                        "Nom fichier": image_path,
                        "Code Affaire": "Erreur lecture fichier",
                        "Nom Client": "",
                        "Prestation": "",
                        "Prestation nom complet": "",
                        "Nombre d'unitée": "",
                        "Surface (m2 ou ml)": "",
                        "Prix Unité": "",
                        "Prix préstation": ""
                    }
            toutes_les_prestations.append(ligne)

        except Exception as e:
            print(f"\n Erreur: {e}")
            import traceback
            traceback.print_exc()


                
    df = pd.DataFrame(toutes_les_prestations)
    df.to_excel('Recapitulatif_Devis' + str(kernel) + '.xlsx', index=False, engine='openpyxl')
    print(f"Fichier Excel créé avec succès")