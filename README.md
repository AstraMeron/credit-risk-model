# credit-risk-model
## 1. Credit Scoring Business Understanding

### 1.1 The Influence of Basel II on Model Interpretability

The Basel II Capital Accord (and similar regulatory frameworks) heavily influences the model design by demanding rigorous risk measurement and effective model governance.

* **Regulatory Imperative:** Credit scoring models are not merely predictive tools; their output is used for critical financial decisions, including the setting of credit limits and determining the bank's necessary regulatory capital. This high-stakes application necessitates stringent oversight.
* **Explainability and Transparency:** Compliance requires that lending decisions based on the credit score be explainable, transparent, and fair. Bati Bank must be able to articulate the decision logic to both consumers (e.g., explaining why a loan was denied) and regulators/auditors (explaining the process and assumptions underlying the score). A model that cannot be easily audited or explained creates regulatory risk.
* **Model Governance:** The model must operate within an effective model governance framework. This includes demonstrating the model’s conceptual soundness, conducting regular back-testing, and managing model risk. An interpretable model (like Logistic Regression) inherently supports this framework better than an opaque one.

### 1.2 Necessity and Risks of the Proxy Variable

**Why is creating a proxy variable necessary?**

The necessity of a proxy variable stems directly from the use of alternative data (eCommerce transactions) to score borrowers who may have limited or no traditional credit bureau information. In the absence of sufficient historical credit information to define a verifiable default, a credit score cannot be generated. Therefore, the proxy variable serves as a substitute target label, allowing us to translate observable customer behavior (like Recency, Frequency, and Monetary value) into a measurable risk category (is_high_risk) that can be used to train a supervised machine learning model.

**What are the potential business risks of making predictions based on this proxy?**

The primary risk associated with using any proxy variable is Proxy Error, where the behavior proxy does not perfectly correlate with the true default event. This leads to several business and ethical risks:

1.  **Financial Loss (False Positives):** The proxy may incorrectly identify an actually high-risk customer as low-risk. Granting credit in this scenario leads to increased loan losses and financial risk for Bati Bank.
2.  **Lost Business (False Negatives):** The proxy may incorrectly flag a creditworthy customer as high-risk, leading to the rejection of profitable business and reduced market share.
3.  **Fairness and Discrimination:** Algorithms developed on non-traditional data may inadvertently learn and perpetuate historical or demographic bias, leading to concerns about discrimination against minority groups.
4.  **Data Accountability:** As the model relies on third-party (eCommerce) data, the bank needs clear audit trails and strong data accountability practices to ensure data quality and privacy.

### 1.3 Trade-offs in Model Selection

In a regulated financial context, Bati Bank must balance the superior predictive power of modern methods against the regulatory demand for transparency.

| Feature | Simple, Interpretable Models (e.g., Logistic Regression with WoE) | Complex, High-Performance Models (e.g., Gradient Boosting) |
| :--- | :--- | :--- |
| **Performance** | Generally lower accuracy, as they often assume linear relationships in the data. | Often provide improved accuracy by capturing complex, non-linear patterns. |
| **Interpretability** | High Transparency. Coefficients (or WoE scores) directly explain risk factors, making decisions easily defensible to consumers and regulators. | Low Transparency ("Black Box"). The relationship between features and the score is opaque, making it difficult to provide simple explanations for decisions. |
| **Regulatory Risk** | Low. Easily compliant with Basel II's governance requirements. | High. Creates the risk of lack of transparency and challenges in satisfying auditors and supervisors. |
| **Conclusion** | The model's selection requires regulators and the bank to strike a balance between innovation and risk. While complex models offer a "more complete picture of borrowers," the necessity of transparency often favors the use of simpler, transparent models like Logistic Regression for core risk calculation. |

---
