# Load type classification

Predict `Load_Type` (`Light_Load`, `Medium_Load`, `Maximum_Load`) from one year of 15-minute industrial electricity readings. The work follows the brief: preprocess the data, explore it, build features, select a model, and evaluate on the last month.

The analysis is in `notebooks/load_type_classification.ipynb`. Run all cells from the top. The notebook finds `data/load_data.csv` whether it is started from this folder or from `notebooks/`.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/load_type_classification.ipynb
```

## Result

The selected model is histogram gradient boosting. It was chosen because it had the best macro-F1 on November (0.600), which is the last month inside the training period. December was not used for that choice.

On December 2018 (2,976 intervals):

| Metric | Light_Load | Medium_Load | Maximum_Load | Overall |
| --- | ---: | ---: | ---: | ---: |
| Precision | 0.947 | 0.991 | 0.979 | |
| Recall | 0.990 | 0.898 | 0.953 | |
| F1-score | 0.968 | 0.942 | 0.965 | macro 0.958, weighted 0.961 |
| Accuracy | | | | 0.962 |

Always predicting light load scores 0.586 accuracy and 0.246 macro-F1 on the same month.

November is the hard month. Weekday peak hours change with the season: January puts maximum load in the late afternoon and evening, and October puts it in the early afternoon. November is the first return to the winter timetable, so a model trained through October is applying the fall hours. December follows November, which is why the December score is back in line with September and October once November is in the training set.

## What the cleaning does

- Sorts the file. Midnight is stored after 23:45, so the raw file steps backward once a day.
- Drops the conflicting duplicate at 31 December 2018 20:00 and keeps the row whose usage and power factor are consistent.
- Rebuilds `NSM` from the timestamp. 3,869 original values were blank or disagreed with the clock.
- Rebuilds power factor from usage and reactive power. Stored values above 100, up to 300, are replaced.
- Fills the remaining blanks with the January–November median. Zero is left as a real reading, not used as a fill.
