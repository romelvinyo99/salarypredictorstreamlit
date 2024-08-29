def main():
    import math
    import pandas as pd
    import streamlit as st
    from CleanerPage import Cleaner
    from Exploratory import EDA, Plot
    from HomePage import Home
    from StatsHyperTune import HyperParameterTuning, StatsModelling, DeepLearning

    # Page 1 - Homepage
    st.set_page_config(
        page_title="Salary Predictor",
        page_icon="statistics.png"

    )
    options = st.sidebar.selectbox(
        "Sections",
        [] + ["Home Page", "Data Cleaning", "Data analysis and visualization", "Statistical Hyper Parameter Tuning",
              "Statistical Modelling and prediction",
              "Deep Learning modelling and Prediction"])
    st.sidebar.success("Select an option above.")
    if st.sidebar.button("Sign out"):
        st.session_state.login_success = False
        st.session_state.tries = 4
    # DataFrames
    cleaned_data = Cleaner("salaryData.csv")
    clean_data = cleaned_data.finalResult()
    eda = EDA(clean_data)
    eda_data, scaler = eda.finalfinal()
    plot = Plot(eda_data)
    anomalied_data = plot.finalResult()
    plot = Plot(eda_data)

    hyper_df = HyperParameterTuning(anomalied_data)
    clustered_data = hyper_df.copy
    if options == "Deep Learning modelling and Prediction":
        st.markdown("---")
        st.header("Deep Learning Modelling and Prediction")
        deep = DeepLearning(clustered_data)
        age, level, experience = StatsModelling.inputField()
        age, level, experience = StatsModelling.userScaling(scaler, age, level, experience)
        estimator = deep.model()
        predicted_salary = deep.predict(age, level, experience, estimator)
        st.write("\n\n\n")
        st.success(f"Predicted salary = {predicted_salary[0][0]}ksh")
    if options == "Home Page":
        raw_df = pd.read_csv("salaryData.csv")
        home = Home(raw_df)
    if options == "Data Cleaning":
        cleaner = Cleaner("SalaryData.csv")
        cleaner.dropNullValues()
        cleaner.dropDuplicates()
    if options == "Data analysis and visualization":
        cleaned_data = Cleaner("salaryData.csv")
        clean_data = cleaned_data.finalResult()
        eda = EDA(clean_data)
        eda.labelling()
        eda.summaryStats()
        eda.correlationMatrix()
        eda.scaling()
        eda.droppingUnnecessaryColumn()
        eda.finalResult()
        plot = Plot(eda_data)
        plot.scatterplot()
        plot.anomalyDetection()
        anomalied_data = plot.finalResult()
    if options == "Statistical Hyper Parameter Tuning":
        hyper = HyperParameterTuning(anomalied_data)
        hyper.linearRegressionModels()
        hyper.backdoor()

    if options == "Statistical Modelling and prediction":
        st.markdown("---")
        st.header("Statistical Modelling and prediction")
        st.markdown("---")
        models = st.selectbox(
            "Choose Model",
            [] + ["Lasso Regression", "Ridge Regression", "Linear Regression", "Random Forest Regression"]
        )
        if models == "Linear Regression":
            model = StatsModelling(clustered_data)
            estimator = model.linearRegression()
            age, level, experience = model.inputField()
            age, level, experience = model.userScaling(scaler, age, level, experience)
            predicted_salary = model.predict(age, level, experience, estimator)
            st.success(f"Predicted salary = {predicted_salary[0]}")
        if models == "Ridge Regression":
            model = StatsModelling(clustered_data)
            estimator = model.ridgeRegression()
            age, level, experience = model.inputField()
            age, level, experience = model.userScaling(scaler, age, level, experience)
            predicted_salary = model.predict(age, level, experience, estimator)
            st.success(f"Predicted salary = {predicted_salary[0]}ksh")
        if models == "Lasso Regression":
            model = StatsModelling(clustered_data)
            estimator = model.lassoRegression()
            age, level, experience = model.inputField()
            age, level, experience = model.userScaling(scaler, age, level, experience)
            predicted_salary = model.predict(age, level, experience, estimator)
            st.success(f"Predicted salary = {math.floor(predicted_salary[0])}ksh")

        if models == "Random Forest Regression":
            model = StatsModelling(clustered_data)
            estimator = model.randomForestRegression()

            age, level, experience = model.inputField()
            age, level, experience = model.userScaling(scaler, age, level, experience)
            predicted_salary = model.predict(age, level, experience, estimator)
            st.write("\n\n\n")
            st.success(f"Predicted salary = {math.floor(predicted_salary[0])}ksh")




