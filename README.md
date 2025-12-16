# Credit Risk Scoring System - Bati Bank

A complete end-to-end credit risk modeling pipeline that translates eCommerce transaction data into actionable credit scores. This project covers data engineering, model training with MLflow tracking, API deployment with FastAPI, and automated CI/CD.

## 1. Credit Scoring Business Understanding

### 1.1 The Influence of Basel II on Model Interpretability

The Basel II Capital Accord (and similar regulatory frameworks) heavily influences the model design by demanding rigorous risk measurement and effective model governance.

* **Regulatory Imperative:** Credit scoring models are not merely predictive tools; their output is used for critical financial decisions, including the setting of credit limits and determining the bank's necessary regulatory capital.
* **Explainability and Transparency:** Compliance requires that lending decisions based on the credit score be explainable, transparent, and fair. Bati Bank must be able to articulate the decision logic to both consumers and regulators.
* **Model Governance:** The model must operate within an effective framework. An interpretable model (like Logistic Regression or tuned Random Forests) supports this framework better than an opaque one.

### 1.2 Necessity and Risks of the Proxy Variable

**Why is creating a proxy variable necessary?**
In the absence of sufficient historical credit information to define a verifiable default, the proxy variable serves as a substitute target label. It allows us to translate observable customer behavior (like Recency, Frequency, and Monetary value) into a measurable risk category (`is_high_risk`) to train a supervised model.

**What are the potential business risks of making predictions based on this proxy?**
1. **Financial Loss (False Positives):** The proxy may incorrectly identify an actually high-risk customer as low-risk.
2. **Lost Business (False Negatives):** The proxy may incorrectly flag a creditworthy customer as high-risk.
3. **Fairness and Discrimination:** Algorithms may inadvertently learn and perpetuate historical or demographic bias from non-traditional data.
4. **Data Accountability:** As the model relies on third-party eCommerce data, the bank needs clear audit trails for data quality and privacy.

### 1.3 Trade-offs in Model Selection
| Feature | Simple Models (e.g., Logistic Regression) | Complex Models (e.g., Gradient Boosting) |
| :--- | :--- | :--- |
| **Performance** | Generally lower accuracy (linear assumptions). | Higher accuracy (captures non-linear patterns). |
| **Interpretability** | High Transparency. Coefficients directly explain risk. | Low Transparency ("Black Box"). |
| **Regulatory Risk** | Low. Easily compliant with Basel II. | High. Challenges in satisfying auditors. |

---

## 2. Technical Architecture

### 2.1 Model Registry (MLflow)
The project uses **MLflow** for experiment tracking and model versioning.
* **Tracking:** Every training run logs hyperparameters and metrics (ROC-AUC, F1-Score).
* **Registry:** The best performing model is registered as `Credit_Risk_Model` in a centralized SQLite database (`mlflow.db`).

### 2.2 Prediction API (FastAPI)
A RESTful API serves real-time predictions. 
* **Endpoints:**
    * `GET /`: Health check and model status.
    * `POST /predict`: Accepts customer features and returns a risk probability and classification.
* **Validation:** Uses **Pydantic** models to ensure data integrity and provide automatic Swagger documentation (`/docs`).

### 2.3 DevOps & CI/CD
* **Containerization:** A `Dockerfile` and `docker-compose.yml` allow the entire system to be deployed consistently across environments.
* **GitHub Actions:** An automated CI pipeline runs on every push:
    * **Linter (flake8):** Ensures code quality and syntax standards.
    * **Unit Tests (pytest):** Validates data processing logic and model availability.

---

## 3. How to Run


1. **Clone the Repository:**
    ```bash
    git clone [https://github.com/AstraMeron/credit-risk-model](https://github.com/AstraMeron/credit-risk-model)
    cd credit-risk-model
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the API:**
    ```bash
    uvicorn src.api.main:app --reload
    ```

4. **Build and Start with Docker:**
    ```bash
    docker-compose up --build
    ```

5. **Access Swagger UI Documentation:**
    ```text
    [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
    ```

6. **Access MLflow UI:**
    ```bash
    mlflow ui
    ```