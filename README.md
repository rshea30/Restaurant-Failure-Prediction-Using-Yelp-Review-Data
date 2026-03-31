# Restaurant-Failure-Prediction-Using-Yelp-Review-Data
Overview
This project uses machine learning to predict restaurant failure with 88-95% precision by analyzing Yelp review patterns across three distinct markets: Philadelphia (urban), Tampa (tourism), and Santa Barbara (coastal). The study challenges conventional wisdom by demonstrating that review recency matters far more than review quality in predicting business outcomes.

Key Findings
The Engagement Paradox: Time since last review shows a 0.73 correlation with failure, while average star rating shows only -0.05 correlation (nearly zero). Failed restaurants maintain 3.5+ average star ratings, proving that customer satisfaction ≠ financial viability. The key insight: "Restaurants fail in silence, not in bad reviews."
High Predictive Accuracy: Three models (Logistic Regression, Random Forest, Gradient Boosting) achieve 88-95% precision across all cities, with Random Forest providing the best precision-recall balance (88% precision, 92% recall).
Cross-Market Generalizability: The Tampa-trained model transfers exceptionally well to other cities (95% average precision), while the Santa Barbara model struggles (76% precision), demonstrating that mid-sized tourism markets produce the most generalizable failure predictors.

Dataset
9,579 restaurants across three cities
1.15M reviews from Yelp Open Dataset
~40% failure rate (consistent across all markets)

Methodology
Engineered 7 time-series features per restaurant: time since last review, average rating, rating variance, review frequency, review frequency variance, rating trend, and total reviews
Trained separate models for each city plus a combined dataset with 80/20 stratified splits
Evaluated on precision to minimize false alarms for viable businesses
Conducted cross-city transfer learning analysis to test generalizability

Practical Applications
Early Warning System: 3-month review gap = actionable red flag
Investor Risk Assessment: Review recency outperforms traditional financial metrics
Platform Integration: Framework suitable for real-time monitoring dashboards on Yelp/Google

Impact
This research provides a data-driven early warning system that could help restaurant owners, investors, and lenders identify at-risk businesses months before closure, potentially enabling interventions that improve survival rates in an industry with a 40% failure rate.

Technologies: Python, Scikit-learn, Pandas, XGBoost, Matplotlib, Seaborn
