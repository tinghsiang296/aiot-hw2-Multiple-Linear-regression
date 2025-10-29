# Ames Housing Linear Regression Analysis (CRISP-DM)

## Business Understanding
- **Goal**: Predict Ames, Iowa housing sale prices using structural and neighborhood attributes.
- **Success Criteria**: Deliver a linear regression model with strong out-of-sample performance and interpretable feature impacts for pricing guidance.

## Data Understanding
- **Rows × Columns**: 2930 × 82.
- **Target (`SalePrice`)**: mean $180,796, median $160,000, std $79,887.
- **Top correlated features**: SalePrice (1.00), Overall Qual (0.80), Gr Liv Area (0.71), Garage Cars (0.65), Garage Area (0.64), Total Bsmt SF (0.63), 1st Flr SF (0.62), Year Built (0.56), Full Bath (0.55), Year Remod/Add (0.53)
- **High-missing columns removed (>40% missing)**: Pool QC, Misc Feature, Alley, Fence, Mas Vnr Type, Fireplace Qu

## Data Preparation
- Numeric features: 36, categorical features: 37.
- Imputation: median for numeric, most-frequent for categorical.
- Scaling: standardisation applied to numeric features.
- Encoding: one-hot encoding with first level dropped per categorical feature.
- Feature selection: SelectKBest (`f_regression`) retaining top 40 predictors.

## Modeling & Evaluation
- Train/test split: 80% train / 20% test (random_state=42).
- Model: Ordinary Least Squares linear regression.
- Test R²: 0.865
- Test RMSE: $32,924
- Test MAE: $20,661
- CV (5 folds) R² mean ± std: 0.833 ± 0.059
- CV RMSE mean ± std: $32,050 ± $6,196
- Residual diagnostics (test set): mean 2946.73, std 32820.00, skew -0.57, kurtosis 26.42, 5th pct -35595.28, 95th pct 41884.13.

## Feature Insights
- **Top selected features (by F-score)**:
  - Overall Qual: score 4030.76, p-value 0.000e+00, coefficient 20,759.40
  - Gr Liv Area: score 2229.04, p-value 0.000e+00, coefficient 23,953.56
  - Garage Cars: score 1661.54, p-value 5.402e-275, coefficient 5,335.59
  - Garage Area: score 1566.51, p-value 9.043e-263, coefficient 2,683.92
  - Total Bsmt SF: score 1403.28, p-value 4.640e-241, coefficient 3,406.96
  - 1st Flr SF: score 1369.42, p-value 1.939e-236, coefficient 585.06
  - Exter Qual_TA: score 1213.21, p-value 1.492e-214, coefficient -19,449.23
  - Year Built: score 991.67, p-value 8.387e-182, coefficient 3,382.75
  - Full Bath: score 974.44, p-value 3.643e-179, coefficient 229.74
  - Kitchen Qual_TA: score 867.48, p-value 1.787e-162, coefficient -22,890.89
  - Year Remod/Add: score 857.30, p-value 7.407e-161, coefficient 5,197.30
  - Foundation_PConc: score 828.10, p-value 3.452e-156, coefficient 3,949.63
  - Garage Finish_Unf: score 809.72, p-value 3.147e-153, coefficient 49.37
  - Garage Yr Blt: score 778.03, p-value 4.406e-148, coefficient -983.01
  - Bsmt Qual_TA: score 773.92, p-value 2.071e-147, coefficient 2,706.83
- **Largest positive coefficients**: Neighborhood_NoRidge (+42,030.7), Neighborhood_NridgHt (+25,344.5), Gr Liv Area (+21,769.4), Overall Qual (+20,266.7), Bsmt Exposure_Gd (+16,577.1), Sale Type_New (+13,459.3), Exterior 2nd_VinylSd (+6,679.8), BsmtFin SF 1 (+6,523.1), Roof Style_Hip (+6,312.1), Year Remod/Add (+5,784.2)
- **Largest negative coefficients**: TotRms AbvGrd (-1,320.6), Lot Shape_Reg (-3,811.0), Exterior 1st_VinylSd (-4,824.8), Bsmt Exposure_No (-5,155.6), Heating QC_TA (-5,375.0), MS Zoning_RM (-10,945.3), Exter Qual_Gd (-17,830.5), Exter Qual_TA (-19,290.4), Kitchen Qual_Gd (-22,931.3), Kitchen Qual_TA (-24,010.0)

## Prediction Intervals
- Generated 95% confidence and prediction intervals for hold-out predictions to quantify uncertainty bands.

## Recommendations
- Use predicted price with interval bounds to communicate pricing risk and upside for stakeholders.
- Investigate influential outliers (noted heavy tails in residuals) to refine fit or segment the market.
- Extend with interaction terms or non-linear models to capture remaining variance while benchmarking against this interpretable baseline.