# Plant Health Assessment Using Machine Learning

A web-based machine learning prototype that analyzes uploaded leaf images and provides a plant-health assessment alongside optional soil nutrient and condition screening.

## Project Overview

The application combines image classification with a separate, rule-based soil screening module. Users can upload a leaf photograph and enter available soil readings to obtain a consolidated report.

### Main Features

- Leaf image upload and preview.
- Leaf-image verification before classification.
- MobileNetV3-Small image classification.
- Four supported image classes:
  - Healthy
  - Nitrogen_Deficiency
  - Phosphorus_Deficiency
  - Potassium_Deficiency
- Optional nitrogen, phosphorus, potassium, pH and soil-moisture inputs.
- Rule-based soil screening and recommendations.
- A report that keeps image predictions separate from soil-based findings.
- Optional local Ollama integration, if configured.

## Technology Stack

- Python
- Streamlit
- PyTorch
- Torchvision
- MobileNetV3-Small
- Pillow
- NumPy and other project dependencies
- Optional Ollama for local language-model explanations

## Model Evaluation

A candidate model was evaluated on a held-out test set of 56 images.

| Metric | Result |
|---|---:|
| Correct predictions | 54 / 56 |
| Test accuracy | 96.43% |
| Healthy recall | 90.91% |
| Nitrogen deficiency recall | 100% |
| Phosphorus deficiency recall | 100% |
| Potassium deficiency recall | 91.67% |

The earlier production checkpoint achieved 35/56 (62.50%) on the same test set.

**Important:** This is a small test set and does not establish real-world field accuracy. Results may differ on photographs of plants taken under different conditions.

The candidate checkpoint was evaluated separately. Its permanent promotion and full integration with the leaf-verification pipeline have not been confirmed. The verifier may still use the original checkpoint.

## How It Works

1. The user uploads a leaf image.
2. The application checks whether the image is suitable for classification.
3. An accepted image is preprocessed and passed to the image classifier.
4. The application displays the predicted class.
5. Optional soil readings are validated and assessed using separate screening rules.
6. The application presents the available image and soil findings in a report.

Soil values do not directly change the image classifier's prediction.

## Requirements

- Python 3.11 is the version used during project development.
- Git
- A compatible environment with the packages listed in `requirements.txt`.
- The required model checkpoint and its matching metadata.

## Installation

Clone the repository:

```bash
git clone https://github.com/CoolMoon003/PlantHealthWebsite.git
cd PlantHealthWebsite
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is absent or incomplete, install the dependencies specified by the project's actual source code and environment.

## Model Files and Dataset

Model checkpoints and datasets may be excluded from Git because of size, licensing, or repository-management considerations.

Before running the application, ensure that the required checkpoint, class metadata, preprocessing configuration and any other required model files are available in the locations expected by the application.

Do not assume the candidate checkpoint is the application's default model. Check the configured paths before running an evaluation.

The full dataset is not included in this repository unless explicitly added and its redistribution is permitted.

## Run the Application

From the project root, activate the virtual environment and run:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your terminal.

If the application requires model files or environment variables, configure those according to the source code before starting it.

## Limitations

- Image classification is limited to the four supported classes.
- The held-out test set contains only 56 images.
- Test accuracy should not be interpreted as guaranteed real-world performance.
- Soil screening thresholds are provisional and are not universal agricultural recommendations.
- Image classification and soil screening are separate processes.
- Complete candidate-model integration with the verifier requires further validation.

## Future Improvements

- Evaluate the model on a larger, independently collected real-world dataset.
- Validate the complete candidate-model and leaf-verification pipeline.
- Improve rejection of unsupported plants and uncertain images.
- Validate soil screening rules against reliable agronomic references.
- Add further testing, model calibration and deployment support.

## References

- PyTorch documentation: https://pytorch.org/docs/stable/
- Torchvision documentation: https://pytorch.org/vision/stable/
- MobileNetV3 paper: https://arxiv.org/abs/1905.02244
- Streamlit documentation: https://docs.streamlit.io/
- Pillow documentation: https://pillow.readthedocs.io/
- Ollama documentation: https://docs.ollama.com/

Dataset source: Add the verified original dataset URL here before publication. Do not substitute an unrelated dataset or claim that a source was used unless confirmed.

## Academic Project

**Project:** Plant Health Assessment Using Machine Learning  
**School:** SCORE  
**Subject:** Machine Learning (ML)  
**Faculty:** Porkodi S  

**Team Members**
- Naveenbabu H — 23MIS0495
- Aishwaraya A — 23MIS0436

This project is an academic prototype and is not a substitute for professional agricultural diagnosis.
