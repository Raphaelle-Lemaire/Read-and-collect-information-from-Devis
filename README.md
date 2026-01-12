# Read and collect informations from Devis
In this repository, we want to automate the extraction of text from a quote using an OCR mechanism, then we extract important information such as the client's name, services, the cost of each service, and then put it into an Excel file.

## Detailed Features
1. Document Processing
We support to format type, "png" and "pdf". For "pdf", we use [ORB (oriented BRIEF)](https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#details) feature to determine if subsequent pages in a pdf are a continuation of the first pdf quote, a new quote or an other type of document like explaination of the right of the user.
We implements grayscale conversion, Otsu thresholding, and morphological operations to remove noise and tables lines, making text more readable for the OCR engine.

## Smart Information Extraction

We leverages YOLOv8 to identify the structure and extract the text location in the quote. After, we applied [pytesseract](https://pypi.org/project/pytesseract/) to extract all the text select by YOLOv8 (see image).
<img width="2480" height="3505" alt="binary" src="https://github.com/user-attachments/assets/d9654c26-3fa3-4703-82ae-d3069bce7785" />

We automatically identifies the Client Name, Project Code and service provision with the json extract by pytesseract. For name of provision, we try to standardized it with categories name like "Taille de haies" (Hedge trimming) or "Entretien Pelouse" (Lawn maintenance) and we specified that it's possible to have more than one provision for one price. We also extracts quantities, surfaces (m2), and linear measurements (ml) when they exist.


## Installation & Dependencies
Python: 3.9
Install the specific versions used in development:
`` pip install opencv-python==4.8.0.74 numpy==1.24.3 pillow pytesseract==0.3.10 ultralytics torch==2.0.1 openpyxl pdf2image pandas matplotlib ``

## Usage
- Place your quotes in your input directory (e.g., 'Devis/png/').
- Configure the dossier variable in ExtractDataDevis.py (line 658).
- Run the script: python ExtractDataDevis.py.
- The script will iterate through all files in your directory and generate a Excel summary.

## Excel Output
The final Recapitulatif_Devis25.xlsx provides a structured view:

| Nom fichier | Code Affaire | Nom Client | Prestation | Prestation nom complet | Nombre d'unitée	| Surface (m2/ml) | Prix Unité | Prix prestation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Devis annuels 2024 acceptés/pdf/12345 EVA12345.png | EVA12345	| MAIRIE |	Taille de haies	| forfait taille de l'ensemble des haies en fond de cours |
 1	| 150m2	| 45.00	| 45.00


## State of the Art
This project implements methodologies inspired by publication in document information extraction (DIE):

- A-Sawaareekun, Chompunut, and Rajalida Lipikorn. "Menu item extraction from Thai receipt images using deep learning and template-based information extraction." Proceeding of the 2024 6th International Conference on Information Technology and Computer Communications. 2024.
- Flaute, Dylan M. Template-Based Document Information Extraction Using Neural Network Keypoint Filtering. MS thesis. University of Dayton, 2024.
- Gyawali, Srijan, et al. "Automating Document Workflows with ResNet-50 and Template-Based OCR." Journal of Engineering Issues and Solutions 4.1 (2025): 478-485.
