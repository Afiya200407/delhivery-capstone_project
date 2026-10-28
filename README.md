# Delhi Very Last-Mile Delay Intelligence Pipeline

## Project Overview

This project builds a reproducible shipment-delay intelligence
pipeline using synthetic logistics data.

The pipeline cleans shipment records, analyses delivery delays,
predicts delivery hours, classifies potentially delayed shipments,
segments delivery hubs, and generates operational recommendations.

## Data findings

1. **Service type:** Express parcels are delayed most often, at a delay rate of 0.4897
   (119 of 243), compared with 0.2056 for Standard and 0.0857 for Economy.
   Source: `outputs/part_a_delay_by_group.csv`.

2. **Hub tier:** The delay rate rises sharply from Tier 1 (0.0400) to Tier 2 (0.2150)
   to Tier 3 (0.4400).

3. **Distance band:** The delay rate climbs with distance: 0.0219 for 0-200 km,
   0.1368 for 201-400 km, 0.4364 for 401-600 km and 0.8523 for 601+ km.

4. **Strongest correlations with delivery_hours:** The three strongest are
   promised_hours (0.8372), distance_km (0.4013) and n_stops (0.3247),
   ranked by absolute correlation, from `outputs/part_a_correlations.csv`.

5. **Log transform:** Declared value is right-skewed: its mean (1533.05) is far above its
   median (1121.5), while after the log transform they are almost equal
   (7.0277 and 7.0220), so the log makes the distribution roughly symmetric.

## Model results (Regression)

1. **Baseline:** Predicting the training mean gives a validation R² of -0.0212 (RMSE 18.8552 hours).
  It explains none of the variation in delivery time, so every model has to beat it.

2. **Winner on validation:** Linear regression had the lowest validation RMSE, 4.8912 hours (R² 0.9313),
  just ahead of gradient boosting (4.9165 hours, R² 0.9306).

3. **Bias and variance:** At depth 1, train_rmse and val_rmse are both
  high and close together (11.6334 and 12.1324) same with the depths 2, 3 showing train_rmse and val_rmse (9.3687and 10.4114) and (7.5295,8.7033) respectively so the tree underfits (high bias). With no depth limit,
  train_rmse 0.0 but val_rmse is 7.789 as well as with the depths 8, 12 with train_rmse and val_rmse (3.336 and 6.8385) and (1.3613,7.4173) respectively so it overfits (high variance).
  The best depth was 6, with val_rmse 6.2796. 

4. **Learning rate:** 0.1 was best (R² 0.9306). 0.01 gave R² 0.7127: each tree corrects only 1% of the remaining
  error, so 100 trees are not enough to reach a good fit and the model underfits (too slow). 1.0 gave R² 0.8440: each tree applies the full correction, so the model overshoots and follows noise instead of settling on the best fit.

5. **Cross-validation:** The ranking disagrees with validation: gradient boosting scored
  0.9352 (std 0.0044) against 0.9261 (std 0.0048) for linear regression. I would ship Linear Regression because of the reason that Linear Regression performs almost as well as Gradient Boosting, with a cross-validation mean R² of 0.9261 compared to 0.9352 and nearly identical validation scores (0.9313 vs. 0.9306). It is also stable across folds, with a low CV standard deviation of 0.0048, easy to explain to planners because each feature has a clear effect, and inexpensive to retrain. Additionally, it achieves a strong test R² of 0.9493, making it a reliable and practical choice. Only this model was scored on the test set (RMSE 4.5235 hours, R² 0.9493), so there is no test figure for gradient boosting.

6. **Final test score:** Linear regression on the test rows: RMSE 4.5235 hours, R² 0.9493.

## Model Results (Classification)

1. **Blind Baseline:**
   The baseline achieved 76.67% accuracy by predicting every parcel as not delayed, but detected none of the 42 delayed parcels (recall = 0). This shows that accuracy alone is insufficient for evaluating a delay-detection model.

2. **Noise Columns:**
   Five bookkeeping columns contributed 22.44% of the Random Forest's total feature importance, yet ranked 8th, 12th, 14th, 16th, and 17th by F-score, all below 5. This suggests the forest sometimes splits on noise, while F-scores help distinguish more informative features. Feature importance alone does not prove predictive signal.

3. **Feature Selection (k):**
   Among the tested values, k = 7 performed best, achieving 83.33% accuracy. Using k = 5 lost useful signals such as rain_flag and sale_event_flag, while larger values introduced mostly noisy features.

4. **Hyperparameter Tuning:**
   The untuned Random Forest achieved the best validation accuracy (0.8333), outperforming Grid Search (0.7750) and Randomized Search (0.7848), despite Grid Search having the best CV accuracy (0.8952). With only 180 validation samples, small metric differences can be unstable, so over-tuning to one validation split should be avoided.

5. **Threshold Selection:**
   A threshold of 0.5 was selected because it achieved 0.8333 recall while maintaining precision of 0.7955. At 0.6, precision increased to 0.8889, but recall fell to 0.7619, below the 0.80 target. At 0.4, recall increased to 0.8810, but precision dropped to 0.7400.

6. **Final Test Results:**
   At the selected threshold of 0.5, the model achieved precision = 0.6889, recall = 0.7561, and F1 = 0.7209. Test recall was below the 0.80 target and lower than validation recall (0.8333). With only 41 delayed test parcels, each outcome changes recall by approximately 2.4 percentage points. This difference may partly reflect sampling variability; the test results should be reported transparently, without retuning on the test set.