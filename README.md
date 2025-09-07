# Curl Type and Density Classifier with Web Interface

This project builds a deep learning model to classify **wavy** and **curly** hair images into the following categories:

- **2A/2B** (Wavy)
- **2C** (Wavy)
- **3A/3B** (Curly)
- **3C** (Curly)

> **Note**: Types 2A and 2B have been combined, as have Types 3A and 3B, due to the subtle differences between them. With a small dataset, separating them would not have been feasible.

---

## Dataset

The model was trained on a **manually curated dataset of 400 images**, with **100 images per class**. Images were sourced from platforms like Pinterest and Google and were cropped for consistency and accuracy.

Despite the small dataset and the nuanced distinctions between hair types, the model achieved:

- **Validation Accuracy**: **83.33%**  
- **Testing Accuracy**: **81.67%**

> ⚠️ The dataset is **not included** in this repository due to copyright restrictions.

---

## Model Architecture

The model is built on **EfficientNetB2**, a pre-trained **Convolutional Neural Network (CNN)**. Hyperparameters such as dropout rate and learning rate were fine-tuned to improve accuracy and generalization.

---

## Installation

You **do not need to install anything locally**. This project is designed to run on **Google Colab**.

---

## Usage Instructions

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/anisharavishankar04/Curl-Type-Classifier.git
   ```

2. **Open Google Colab**  
   Go to [Google Colab](https://colab.research.google.com/) and sign in.

3. **Open the Notebook**  
   - Go to `File` → `Open notebook` → `Upload`.
   - Upload the `.ipynb` notebook file from the cloned repository.

   > ⚠️ Only the `.ipynb` notebook is provided in the repository. The trained model (`.pth` file) is **not included**. You will need to **run the notebook to generate and save** the model file.

4. **Train and Save Model**  
   - After training, the notebook will save the model as a `.pth` file.
   - Make sure to download and save the file as:  
     ```
     best_model.pth
     ```

5. **Prepare Your Dataset**  
   - Create your own image dataset.
   - Upload it to your **Google Drive** (follow notebook instructions for path structure).

6. **Set Up the Runtime**  
   - Go to `Runtime` → `Change runtime type`.
   - Select **GPU** and choose **T4 GPU** from the dropdown.

7. **Run the Notebook**  
   - Go to `Runtime` → `Run all`.
   - Connect your Google Drive when prompted.

8. **Complete Execution**  
   - Continue running the remaining cells.
   - The training and evaluation will complete automatically.

---

## Notes

- Ensure your dataset follows the folder structure expected by the notebook.
- You may need to fine-tune paths and batch sizes if using a dataset of different size.
