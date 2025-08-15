# Natural-Language Scene Localization (NL Scene Localization)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Project Overview

This project provides a **minimal, dependency-light baseline for natural-language scene localization**.  
It allows you to detect objects and interactions in images based on **textual queries**, leveraging:  

- **OWLv2** for object detection  
- **CLIP** for ranking candidate regions  

The pipeline handles query variants, person-object interactions, person-person interactions, and uses soft-NMS to remove duplicate or overlapping detections.  

---

## Folder Structure

aims/
├── 1.py # Main script
├── snatching.jpg.webp # Example image(s) for testing
└── README.md # This file

---

## Features

- Detect persons and objects in images  
- Build candidate regions:
  - Single objects  
  - Person-object interactions  
  - Person-person interactions  
- Apply **soft-NMS** to refine overlapping detections  
- Rank candidates using **CLIP** similarity with the query  
- Output visualizations and JSON reports  

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AnushkaShukla16/detection_expert.git
cd detection_expert/aims


# Install dependencies
pip install torch torchvision transformers pillow numpy
Usage
Edit 1.py to set your image path and query:
IMAGE_PATH = "snatching.jpg.webp"
QUERY = "men snatching a chain"
Run the script:
python 1.py
Outputs are saved in output_<timestamp>/:
candidates.jpg → All candidate boxes drawn
final.jpg → Top candidate box
final_crop.jpg → Cropped image of the top candidate
report.json → JSON summary of detections and rankings
How It Works
Query Tokenization:
Expands natural language queries into variants and maps synonyms (e.g., "man" → "person").
Object Detection (OWLv2):
Detects objects and persons in the image using fast tokenization.
Candidate Building:
Creates candidate regions for:
Single objects
Person-object interactions
Person-person interactions
Soft-NMS:
Refines candidates by suppressing overlapping boxes.
CLIP Ranking:
Ranks candidates according to similarity with the original query.
Visualization & Output:
Annotates images, saves cropped regions, and generates JSON reports.
Example Query
QUERY = "men snatching a chain"
Detects relevant persons and objects
Highlights candidate boxes on the image
Ranks regions based on CLIP similarity
Dependencies
Python 3.8+
PyTorch
Transformers (Hugging Face)
Pillow
NumPy
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements
OWLv2 for object detection
CLIP for ranking
Inspired by the Visual Genome dataset