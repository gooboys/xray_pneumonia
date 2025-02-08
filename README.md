# **Pneumonia Diagnosis via X-Ray Image Classification**

This repository explores the potential of deep learning in classifying pneumonia from X-ray scans. The models classify images into three categories: **no infection, bacterial infection, and viral infection**. Additionally, we investigate **explainable AI (XAI)** using **Class Activation Maps (CAMs)**.

## **Performance Summary**

- **Infection vs. No Infection:** **0.99 AUC-ROC**
- **Three-Class Classification (No Infection, Bacterial, Viral):** **0.93 AUC-ROC**
- **Bacterial vs. Viral Classification:** **0.88 AUC-ROC**

---
## **Requirements**

This project requires the following dependencies:

- **Python 3.9.20** (Recommended)
- **PyTorch**
- **NumPy**
- **Pandas**
- **OpenCV**
- **Matplotlib**

To set up the environment, you can create a virtual environment and install the required dependencies:

```bash
python3.9 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Then, install dependencies:

```bash
pip install -r requirements.txt
```

---
## **Project Structure**

### **Models**

There are **two primary models** and **two model containers** (one for each approach):

1. **First Model (Single-Model Ensemble)**

   - A bagging ensemble of a single deep neural network structure.
   - Performs **three-class classification** (no infection, bacterial, viral).

2. **Second Model (Two-Step Binary Classification)**

   - A pipeline of two separate binary classifiers:
     - **First stage:** Classifies **infection vs. no infection**.
     - **Second stage:** If infection is present, classifies **bacterial vs. viral**.
   - Uses the **same deep learning model** for the first stage but with a **two-node output layer**.
   - Utilizes **three different transfer learning models** for the second-stage classification:
     - `denseA3` → **DenseNet-121**
     - `denseB5` → **DenseNet-169**
     - `eff4` → **EfficientNet-B0**

### **Branches**

- **`main`**\*\* branch:\*\* Contains the finalized, core code for running models.
- **`container`**\*\* branch:\*\* Holds additional files, experimental models, and unsuccessful approaches not included in `main`.

---

## **Running the Project**

To execute the models, follow this sequence:

1. **`analysis.py`** *(Optional)*

   - Analyzes the dataset.

2. **`preprocessing.py`**

   - Prepares the dataset by generating a CSV file with image paths and labels.
   - **Optimized for efficiency** by avoiding storing all images in memory.

3. **`ensemble_model.py`**

   - Trains the first ensemble model.

4. **`run_ensemble.py`**

   - Runs the first ensemble model and generates **Class Activation Maps (CAMs).**

5. **`infection_present.py`**

   - Runs the **first binary classification** of the second ensemble model (infection presence detection).

6. **`threshholding.py`**

   - Runs the **second binary classification** of the second ensemble model (bacterial vs. viral detection).

7. **`run_full_ensemble.py`**

   - Executes the entire **two-step ensemble model**.

---

## **Supplementary Files**

These files were used for experimentation, debugging, or validation but are **not required** to run the final models.

- **`Bayesian_optimization.py`**

  - Tuned **learning rate, batch size, and dropout rate** using Bayesian optimization.
  - Results were used for final model hyperparameters.

- **`Bayesian_threshhold.py`**

  - Similar to the above but included **threshold rate** in optimization.
  - Did not yield valuable results.

- **`camtest.py`**

  - Directly generates **Class Activation Maps (CAMs)** from a selected model.

- **`ensemble_validation.py`**

  - Developed and validated the **sequential binary classification approach**.
  - Generated accuracy, F1-score, recall, and other metrics.

- **`evaluate_models.py`**

  - Evaluated trained models without retraining.
  - Ensured **no data leakage** (after mistakenly assuming the model was finalized due to leakage).

- **`infection_type.py`**

  - Early testing of **binary classification of infection types** before adding thresholding, L2 regularization, and other enhancements.

---

## **Additional Experiments (Stored in ****************************`container`**************************** branch)**

Files in the `container` branch include:

- Preprocessing and models for **RGB images**.
- Various **single-layer models** and experiments.
- Alternative **ensemble configurations**.
- Other variations not included in `main`.

The `camModels` folder in `main` contains the **finalized models**. The only finalized model not in `camModels` is `standardCNN.py`. 

---

## **Model & Data Storage Information**

- **Dataset Reference**: The dataset used in this project is the ChestXRay dataset from [Mendely Data](https://data.mendeley.com/datasets/rscbjbr9sj/2). Users must download the dataset separately before running the code.
- **Model files (`.pth`) and CSV datasets are not stored in this repository** due to size constraints and reproducibility considerations.
- To use the models, **train them locally**.
- The **Jupyter Notebook (`analysis.ipynb`) relies on pretrained models and preprocessed datasets**. It does not perform training or preprocessing to reduce computational overhead.

---

## **Conclusion**

This project demonstrates the effectiveness of deep learning in **pneumonia classification** and **XAI visualization**. The best-performing approach leverages **ensemble learning and transfer learning**, achieving strong performance in differentiating **no infection, bacterial, and viral pneumonia**.

