# Student Performance Analysis and Prediction Using AI

**Prepared by:** Bushra Jabeen  
**Program:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026

## Project overview
This beginner-friendly project demonstrates a data analytics and machine-learning workflow using Python. It explores synthetic student-related data, creates charts, and trains a Random Forest model to classify demonstration performance categories.

## Files
- `BushraJabeen_StudentPerformance.py` — Python script
- `BushraJabeen_StudentPerformance.ipynb` — Jupyter Notebook
- `student_performance_dataset.csv` — 600 synthetic demonstration records
- `requirements.txt` — required Python packages
- `BushraJabeen_StudentPerformance_ProjectReport.docx` — project report

## Dataset
The CSV contains study hours, attendance, previous score, assignment completion, sleep hours, an illustrative final score, and a performance category. All rows are synthetic and do not describe real students.

## Run locally
1. Install Python 3.10 or newer.
2. Keep all project files in the same folder.
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python BushraJabeen_StudentPerformance.py
   ```
5. Or open the notebook in Jupyter and run all cells:
   ```bash
   jupyter notebook
   ```

Charts are saved in the `outputs` folder created by the script.

## Method
1. Load and inspect the CSV using Pandas.
2. Check missing values and summary statistics.
3. Visualize performance categories and study hours versus final score.
4. Split the dataset into 80% training and 20% testing data.
5. Train a Random Forest classifier.
6. Evaluate predictions using accuracy, a classification report, and a confusion matrix.
7. Plot feature importance and show a hypothetical example prediction.

## Important limitation
The dataset is synthetic and the target category is derived from an illustrative score rule. The model is for learning and demonstration only. Do not use it to grade, rank, label, or make decisions about real students.

## References
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/
- Scikit-learn: https://scikit-learn.org/stable/
