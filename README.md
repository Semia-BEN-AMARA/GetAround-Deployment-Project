# GetAround - Pricing Optimization & Delay Analysis

This project is part of the **Data Science & Engineering Certification (RNCP6)**. It delivers a comprehensive data-driven solution to optimize GetAround's car-sharing operations, addressing two critical business challenges: 

1. **Late Checkouts Analysis:** A Streamlit dashboard to understand delays and recommend an optimal threshold minimum time window between rentals.
2. **Pricing Optimization:** A Machine Learning model served via a FastAPI endpoint to predict the ideal daily rental price of a car based on its features.

## Project Structure
* `/dashboard`: Contains the Streamlit web application and its deployment files.
* `/api`: Contains the FastAPI code, machine learning training script, and saved model.


## Production Links
* **Interactive Business Dashboard (Streamlit Cloud):** [https://getaround-deployment-project-bpphz8id5sy8rbkvyvezsp.streamlit.app/](https://getaround-deployment-project-bpphz8id5sy8rbkvyvezsp.streamlit.app/)
* **Production API & Documentation (Hugging Face Spaces):** [https://huggingface.co/spaces/semia91/getaround-pricing-api][https://semia91-getaround-pricing-api.hf.space]
* **API Interactive OpenAPI Specs:** [https://semia91-getaround-pricing-api.hf.space/docs]

---

## 📂 Project Architecture

```text
GetAround-Deployment-Project/
├── api/
│   ├── app.py                  # FastAPI implementation (Inference)
│   ├── pricing_model.joblib    # Serialized ML Pipeline
│   ├── requirements.txt        # Production API dependencies
│   └── Dockerfile              # Containerization for Hugging Face Spaces
├── dashboard/
│   ├── app.py                  # Streamlit Dashboard (Delay analysis)
│   ├── requirements.txt        # Dashboard dependencies
│   └── src/
│       ├── train.py            # ML Training pipeline with MLflow tracking
│       ├── ML_analysis.ipynb   # Notebook for Machine Learning exploration
│       ├── exploratory_analysis.ipynb
│       ├── get_around_delay_analysis.xlsx
│       └── get_around_pricing_project.csv
├── .gitignore
└── README.md