# Odin-Mind
# 📘 Math Question Classification using Classical Machine Learning

This project builds a **classical machine learning pipeline** to classify high school mathematics questions into subtopics such as **Algebra, Geometry, Number Theory, Precalculus**, etc.

Each question is stored as an **individual JSON file**, and the dataset is already split into **training** and **testing** sets, organized in **class-wise folders**.

---

## 🎯 Objective

- Classify math questions into their respective subtopics
- Use **classical ML techniques** (no deep learning for classification)
- Maintain interpretability and simplicity
- Ensure no data leakage by using pre-split train/test data

---

## 📂 Dataset Structure
data/
└── raw/
├── train/
│ ├── algebra/
│ ├── geometry/
│ ├── number_theory/
│ └── ...
└── test/
├── algebra/
├── geometry/
├── number_theory/
└── ...


- Each subfolder name represents the **class label**
- Each file is a JSON with at least the following field:

json
{
  "problem": "What is the value of √(3⁵ + 3⁵ + 3⁵)?"
}


