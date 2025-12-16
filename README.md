# Credit Risk Probability Model for Alternative Data

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

## Project Overview

This project implements a credit scoring model using alternative data from an e-commerce platform to enable buy-now-pay-later (BNPL) services. The model transforms behavioral transaction data into predictive risk signals using Recency, Frequency, and Monetary (RFM) analysis.

## Business Context

Bati Bank is partnering with an e-commerce company to provide customers with the ability to buy products on credit. This project creates a Credit Scoring Model using transaction data from the e-commerce platform to assess creditworthiness and predict default risk.

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord fundamentally changed banking regulation by requiring banks to quantify and manage credit risk more rigorously. It introduced three pillars: minimum capital requirements, supervisory review, and market discipline. Under Pillar 1, banks must calculate risk-weighted assets based on their internal risk assessments, which directly impacts the amount of capital they must hold.

**Key Implications for Our Model:**

- **Regulatory Compliance**: Basel II requires banks to demonstrate that their risk models are sound, validated, and well-documented. Regulators need to understand how the model works to approve its use for capital calculations.

- **Model Validation Requirements**: The Accord mandates that models undergo rigorous validation processes. An interpretable model allows regulators and internal audit teams to:
  - Verify that the model captures genuine risk factors
  - Understand the relationship between inputs and outputs
  - Assess whether the model behaves reasonably under different scenarios
  - Identify potential biases or discriminatory practices

- **Capital Adequacy**: Since capital requirements are directly tied to risk estimates, any errors in the model can lead to either:
  - **Under-capitalization**: Insufficient capital buffers, risking regulatory penalties and financial instability
  - **Over-capitalization**: Excess capital that could be deployed more productively elsewhere

- **Transparency and Accountability**: In a regulated environment, stakeholders (regulators, auditors, board members) need to understand model decisions. An interpretable model enables:
  - Clear explanations for credit decisions
  - Identification of key risk drivers
  - Ability to challenge and refine model assumptions
  - Compliance with fair lending practices

- **Documentation Standards**: Basel II requires comprehensive documentation of model development, assumptions, limitations, and performance. An interpretable model is inherently easier to document and justify.

Therefore, our model must not only be accurate but also transparent, explainable, and thoroughly documented to meet regulatory standards and enable effective risk management.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

**Why a Proxy Variable is Necessary:**

In traditional credit scoring, default labels come from historical loan performance data—customers who failed to repay loans within specified timeframes. However, in this project, we're working with e-commerce transaction data where no actual credit products have been issued yet. We have no direct observation of default behavior.

A proxy variable is necessary because:
- **No Historical Credit Data**: The e-commerce platform hasn't offered credit products before, so we have no actual default records
- **Model Training Requirement**: Supervised learning models require labeled data to learn patterns. Without a target variable, we cannot train a predictive model
- **Business Need**: The bank needs to make credit decisions immediately, before accumulating years of default data
- **Alternative Data Opportunity**: We can leverage behavioral patterns (RFM metrics) that correlate with creditworthiness

**Potential Business Risks of Proxy-Based Predictions:**

1. **Proxy Risk (Fundamental Risk)**: 
   - **Risk**: The proxy (e.g., "disengaged customers" = high risk) may not accurately reflect actual default behavior. A customer who rarely shops might still be creditworthy, while an active shopper might default.
   - **Impact**: Model may approve bad customers or reject good ones, leading to financial losses or missed opportunities

2. **Concept Drift**:
   - **Risk**: The relationship between the proxy and actual default may change over time. E-commerce behavior patterns may evolve, making the proxy less predictive.
   - **Impact**: Model performance degrades over time without detection, leading to increasing losses

3. **Selection Bias**:
   - **Risk**: The proxy may be based on a subset of customer behaviors that don't represent the full population of potential borrowers.
   - **Impact**: Model may be biased against certain customer segments, leading to unfair lending practices and potential regulatory issues

4. **Overfitting to Proxy**:
   - **Risk**: The model may learn patterns specific to the proxy definition rather than genuine default risk factors.
   - **Impact**: Poor generalization to actual default scenarios, model fails in production

5. **Regulatory and Legal Risks**:
   - **Risk**: Regulators may question the validity of proxy-based models, especially if they lead to discriminatory outcomes.
   - **Impact**: Model rejection, regulatory penalties, reputational damage

6. **Business Model Misalignment**:
   - **Risk**: The proxy definition (e.g., low RFM = high risk) may not align with the actual business model. For BNPL, customers with low transaction frequency might still be good credit risks if they have stable income.
   - **Impact**: Suboptimal credit decisions, reduced profitability

**Mitigation Strategies:**
- Continuously validate proxy against actual default data as it becomes available
- Use multiple proxies and ensemble approaches
- Implement robust monitoring and model retraining protocols
- Maintain clear documentation of proxy assumptions and limitations
- Regular backtesting and performance monitoring

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

**Simple, Interpretable Models (e.g., Logistic Regression with WoE):**

**Advantages:**
- **Regulatory Compliance**: Easy to explain to regulators, auditors, and stakeholders. Each coefficient has a clear meaning
- **Transparency**: Decision logic is explicit—stakeholders can trace exactly how each feature contributes to the final score
- **Debugging**: Easy to identify and fix issues when the model behaves unexpectedly
- **Fair Lending Compliance**: Can demonstrate that decisions are based on legitimate risk factors, not protected characteristics
- **Stability**: Less prone to overfitting, more stable predictions across different data distributions
- **Documentation**: Straightforward to document and validate
- **Feature Engineering Insight**: WoE transformation provides business insights into feature importance and risk relationships

**Disadvantages:**
- **Performance Limitations**: May not capture complex non-linear relationships and feature interactions
- **Lower Predictive Power**: Often achieves lower AUC/accuracy compared to ensemble methods
- **Manual Feature Engineering**: Requires more domain expertise and manual work to create meaningful features

**Complex, High-Performance Models (e.g., Gradient Boosting/XGBoost):**

**Advantages:**
- **Superior Predictive Performance**: Typically achieves higher AUC, better precision-recall balance, and lower error rates
- **Automatic Feature Interactions**: Can discover complex patterns and interactions without manual engineering
- **Robustness**: Better handling of non-linear relationships and missing data
- **Competitive Advantage**: Better risk discrimination can lead to more profitable lending decisions

**Disadvantages:**
- **Black Box Nature**: Difficult to explain individual predictions or understand feature contributions
- **Regulatory Challenges**: Regulators may be skeptical of models they cannot interpret, potentially delaying approval
- **Fair Lending Risks**: Hard to prove that the model doesn't discriminate against protected groups
- **Debugging Difficulty**: When the model fails, it's challenging to identify root causes
- **Overfitting Risk**: More prone to overfitting, especially with limited data
- **Documentation Burden**: Requires extensive documentation of model architecture, hyperparameters, and validation procedures

**Key Trade-offs in Regulated Financial Context:**

1. **Regulatory Approval vs. Performance**:
   - Simple models are easier to get regulatory approval but may sacrifice predictive power
   - Complex models may face regulatory scrutiny despite better performance

2. **Explainability vs. Accuracy**:
   - Regulators and customers increasingly demand explainable AI
   - Simple models provide clear explanations but may miss important patterns
   - Complex models can use techniques like SHAP values for post-hoc explanation, but this adds complexity

3. **Development Time vs. Performance**:
   - Simple models with WoE can be developed and validated faster
   - Complex models require more time for tuning, validation, and documentation

4. **Maintenance vs. Performance**:
   - Simple models are easier to maintain, update, and retrain
   - Complex models require more sophisticated MLOps infrastructure

5. **Business Trust vs. Performance**:
   - Business stakeholders may trust simple, explainable models more
   - Complex models may be viewed with suspicion, even if they perform better

**Recommended Approach:**

A **hybrid strategy** is often best in regulated contexts:
- Start with a simple, interpretable model (Logistic Regression with WoE) as a baseline
- Use it for regulatory approval and stakeholder buy-in
- Develop a complex model (Gradient Boosting) in parallel for comparison
- If the complex model shows significant performance gains, invest in explainability tools (SHAP, LIME) and comprehensive documentation
- Consider ensemble approaches that combine both model types
- Always maintain the simple model as a benchmark and fallback option

The final choice depends on:
- Regulatory requirements in your jurisdiction
- Available data volume and quality
- Business risk tolerance
- Resources for model maintenance and documentation
- Stakeholder preferences for transparency vs. performance

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Key Features

- **Proxy Target Variable**: RFM-based clustering to identify high-risk customers
- **Feature Engineering**: Automated pipeline with WoE transformation and IV analysis
- **Model Training**: Multiple algorithms with hyperparameter tuning and MLflow tracking
- **Model Deployment**: Containerized FastAPI service
- **CI/CD Pipeline**: Automated testing and code quality checks

## Getting Started

(Instructions will be added as the project progresses)

## References

- [Basel II Capital Accord](https://www.bis.org/publ/bcbs107.htm)
- [Alternative Credit Scoring - HKMA](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Credit Scoring Approaches Guidelines - World Bank](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
