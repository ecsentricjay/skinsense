# SkinSense — Skin Disease Detection for African Skin Types

SkinSense is a decision-support prototype that uses transfer learning (MobileNetV2 pretrained on ImageNet) to classify uploaded skin images into one of eight common dermatological conditions with particular relevance to African skin types: Eczema, Tinea Versicolor, Acne Vulgaris, Vitiligo, Psoriasis, Melanoma, Acne Keloidalis Nuchae, and Seborrheic Dermatitis. The Streamlit interface allows a user to upload a photograph, run inference with a single click, and view the top prediction, a confidence score, a ranked top-3 table, a condition-specific clinical note tailored to darker skin presentations, and a medical disclaimer — making it suitable for academic project demonstrations, medical-informatics coursework, and early-stage clinical decision-support research.

## Installation

```bash
pip install -r requirements.txt
```

> **Windows users:** `requirements.txt` uses `tensorflow-cpu` instead of `tensorflow`.
> TensorFlow dropped native Windows GPU support after v2.10, but `tensorflow-cpu`
> (same API, full Windows builds) is all that is needed for this CPU-only prototype.

## Running the App

```bash
streamlit run app.py
```

The first time the app starts it will automatically run `model.py` to generate `model_architecture.json` and `model_weights.h5` in the current directory (this takes roughly 15–30 seconds on a standard laptop CPU). Subsequent starts load the cached files and are much faster.

## Project Structure

| File | Purpose |
|------|---------|
| `model.py` | Defines and serialises the MobileNetV2-based Keras model |
| `predict.py` | Pre-processing and inference pipeline |
| `app.py` | Streamlit single-page application |
| `requirements.txt` | Python dependencies |
| `model_architecture.json` | *Generated* — Keras model architecture |
| `model_weights.h5` | *Generated* — Model weights |

## Important Note on Model Weights

`model_weights.h5` contains **random initialisation weights** and is intended for demonstration purposes only. Predictions made by this prototype are not clinically meaningful. For a production or research deployment, replace `model_weights.h5` with weights obtained by fine-tuning the model on a properly curated and labelled dermatology dataset (e.g., DermNet, ISIC Archive, or an in-house African skin dataset). The model architecture in `model.py` is designed to be drop-in compatible with such training.

## Disclaimer

This tool is a student prototype and is **not a medical device**. It must not be used for clinical diagnosis. All outputs should be interpreted by a qualified healthcare professional.
"# skinsense" 
