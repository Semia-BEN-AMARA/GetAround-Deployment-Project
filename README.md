# GetAround - Data Product & Deployment Project

## Project Overview
This project focuses on building and deploying a complete data solution for GetAround (a car-sharing platform) to solve two business challenges:
1. **Late Checkouts Analysis:** A Streamlit dashboard to understand delays and recommend an optimal threshold minimum time window between rentals.
2. **Pricing Optimization:** A Machine Learning model served via a FastAPI endpoint to predict the ideal daily rental price of a car based on its features.

## Project Structure
* `/dashboard`: Contains the Streamlit web application and its deployment files.
* `/api`: Contains the FastAPI code, machine learning training script, and saved model.

## Tech Stack
* **Language:** Python 3.10+
* **Dashboard:** Streamlit
* **API:** FastAPI, Pydantic
* **Machine Learning:** Scikit-Learn
* **DevOps:** Docker