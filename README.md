# 🍽️ Food Waste Reduction Analysis

## 📑 Project Overview

This project analyzes global food waste patterns to identify causes and propose solutions. Our key findings show that 1.3 billion tons of food are wasted annually, with an economic loss of $101,560 million. Household consumption accounts for 62% of waste, with prepared foods and fresh produce being major contributors.

## 👥 Team Members

- Hazim Al-Hatim (221110149)
- Elsayed Azab (221110389)
- Abdulrahman Alsaber (221111057)

## 📊 Dataset

Our analysis uses two Kaggle datasets:

- [Global Food Wastage Dataset](https://www.kaggle.com/datasets/atharvasoundankar/global-food-wastage-dataset-2018-2024) (5,002 records)
- [Country-Level Food Waste Research](https://www.kaggle.com/datasets/joebeachcapital/food-waste) (216 countries)

## 🔬 Methods

1. **Data Analysis**: Correlation analysis, sectoral breakdown, and geographic trends
2. **Machine Learning**: Random Forest model achieved 95% R² in predicting waste

## 📈 Key Findings

- Household waste dominates (62%), followed by food service (24%) and retail (14%)
- Economic loss strongly correlates with waste volume (correlation: +0.92)
- Random Forest model showed 95% importance of economic factors in predicting waste

## 💡 Recommendations

| Sector       | Strategies                               | Impact           |
| ------------ | ---------------------------------------- | ---------------- |
| Household    | Education, meal planning, better storage | 25-30% reduction |
| Food Service | Portion control, donation systems        | 20-25% reduction |
| Retail       | Dynamic pricing, improved storage        | 15-20% reduction |

## 🚦 How to Run

```bash
git clone https://github.com/yourusername/food-waste-reduction.git
pip install -r requirements.txt
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`
