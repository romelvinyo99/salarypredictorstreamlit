from io import StringIO
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import pandas as pd


class EDA:
    def __init__(self, cleaned_data):
        self.data = cleaned_data
        self.scaler = None

    def labelling(self):
        st.header("Exploratory Data Analysis and Visualization")
        st.markdown("---")
        copy = self.data
        st.subheader("1.Labelling of categorical data")
        st.subheader("Before labelling")
        st.write(self.data.head(5))
        st.text("""It is essential for labelling of categorical values for better evaluation and interpretation.\n\nThere are various methods:\n
                   \t1.One hot Encoding - label encoder + column transformer\n
                   \t2.Getting Dummy variables\nIn this data set we are not going to consider dummy variable trap hence the use of label encoder standalone 
                """)
        st.markdown("---")
        code = """
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for column in df.select_dtypes(include = ["object"]).columns:
                   df[column] = le.fit_transform(df[column])
                df.head(3)"""
        st.code(code, language="python")
        le = LabelEncoder()
        for column in copy.select_dtypes(include=["object"]).columns:
            copy[column] = le.fit_transform(copy[column])
        st.subheader("After labelling")
        st.write(copy.head(5))

    def correlationMatrix(self):
        st.markdown("---")
        st.subheader("3.Correlation Matrix")
        st.markdown("---")
        st.text(
            "Creation a correlation matrix is important to determine the interdependence of features\nwith the target variable.")
        code = """
                correlation_matrix = df.corr()
                correlation_matrix_salary = correlation_matrix["Salary"]
                correlation_matrix_salary
                """
        st.code(code, language="python")
        correlation_matrix_salary = self.data.corr()["Salary"]
        buffer = StringIO()
        correlation_matrix_salary.to_string(buf=buffer)
        st.text(buffer.getvalue())
        st.text(
            "Gender and Job Title have a very low correlation to salary\n\nTo create a good regression model we will drop this column")

    def droppingUnnecessaryColumn(self):
        st.markdown("---")
        st.subheader("5.Dropping unnecessary columns")
        st.markdown("---")
        st.text("Dropping the columns with least correlation according to karl pearson's coefficient of correlation")
        code = """
                df1 = df.drop(columns = ["Gender", "Job Title"], axis = 1)
                print(df1.head(3))
                """
        st.code(code, language="python")
        copy = self.data
        copy = copy.drop(columns=["Gender", "Job Title"], axis=1)
        st.write(copy.head(3))

    def scaling(self):
        copy = self.data
        le = LabelEncoder()
        for column in copy.select_dtypes(include=["object"]).columns:
            copy[column] = le.fit_transform(copy[column])

        st.markdown("---")
        st.subheader("4.Feature Scaling")
        st.markdown("---")
        st.text(
            """Feature scaling is important for multiple reasons\n\nFaster convergence - Our main model is based on regression this will reduce the number of iterations before convergence to get the coefficients""")
        scaler = MinMaxScaler()
        copy[["Age", "Education Level", "Gender", "Job Title", "Years of Experience"]] = (
            scaler.fit_transform(copy[["Age", "Education Level", "Job Title", "Gender", "Years of Experience"]]))
        st.text(
            "For this dataset we will use Min Max scaling this is because we need not want to make assumptions about the distribution of the data\n\nThis will come in handy when we use isolation forest for anomaly detection")
        code = ("""
                  scaler = MinMaxScaler()
                  df[["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]] = (scaler.fit_transform(df[["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]]))
                """)
        st.code(code, language="python")
        st.write("Output")
        st.write(copy)

    def summaryStats(self):
        st.markdown("---")
        st.subheader("2.Summary statistics")
        st.markdown("---")
        st.write(self.data.describe())

    # Returns labelled - scaled - dropped data
    def finalResult(self):
        copy = self.data
        le = LabelEncoder()
        for column in copy.select_dtypes(include=["object"]).columns:
            copy[column] = le.fit_transform(copy[column])
        copy = copy.drop(columns=["Gender", "Job Title"], axis=1)
        scaler = MinMaxScaler()
        copy[["Age", "Education Level", "Years of Experience"]] = (
            scaler.fit_transform(copy[["Age", "Education Level", "Years of Experience"]]))
        st.markdown("---")
        st.subheader("Summary Information - after Exploratory Data Analysis")
        st.markdown("---")
        st.write(copy.describe())
        return self.data

    def finalfinal(self):
        le = LabelEncoder()
        for column in self.data.select_dtypes(include=["object"]).columns:
            self.data[column] = le.fit_transform(self.data[column])
        self.data = self.data.drop(columns=["Gender", "Job Title"], axis=1)
        scaler = MinMaxScaler()
        self.scaler = scaler
        self.data[["Age", "Education Level", "Years of Experience"]] = (
            scaler.fit_transform(self.data[["Age", "Education Level", "Years of Experience"]]))
        return self.data, self.scaler


class Plot:
    def __init__(self, eda_data):
        self.data = eda_data

    def scatterplot(self):
        st.markdown("---")
        st.subheader("5.Plotting scatter plots between:\nX = Age / Education Level / Job Title\n\ny = Salary")
        # Create a figure with a 1x3 grid of subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        # Scatter plots
        axs[0].scatter(self.data["Age"], self.data["Salary"], color="blue", label="datapoints")
        m1, b1 = np.polyfit(self.data["Age"], self.data["Salary"], 1)
        axs[0].plot(self.data["Age"], m1 * self.data["Age"] + b1, color="red", label="regression line")
        axs[0].set_title('Age vs Salary')
        axs[0].grid()
        axs[0].legend()
        axs[1].scatter(self.data["Education Level"], self.data["Salary"], color="red", label="datapoints")
        m2, b2 = np.polyfit(self.data["Education Level"], self.data["Salary"], 1)
        axs[1].plot(self.data["Education Level"], m2 * self.data["Education Level"] + b2, color="red",
                    label="regression line")
        axs[1].set_title('Education Level vs Salary')
        axs[1].legend()
        axs[1].grid()
        axs[2].scatter(self.data["Years of Experience"], self.data["Salary"], color="green", label="datapoints")
        m3, b3 = np.polyfit(self.data["Years of Experience"], self.data["Salary"], 1)
        axs[2].plot(self.data["Years of Experience"], m3 * self.data["Years of Experience"] + b3, color="red",
                    label="regression line")
        axs[2].legend()
        axs[2].grid()
        axs[2].set_title('Years of Experience vs Salary')
        code = ("""
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Scatter plots
        axs[0].scatter(self.data["Age"], self.data["Salary"], color="blue", label="datapoints")
        m1, b1 = np.polyfit(self.data["Age"], self.data["Salary"], 1)
        axs[0].plot(self.data["Age"], m1 * self.data["Age"] + b1, color="red", label="regression line")
        axs[0].set_title('Age vs Salary')
        axs[0].legend()
        axs[0].grid()
        axs[1].scatter(self.data["Education Level"], self.data["Salary"], color="red", label="datapoints")
        m2, b2 = np.polyfit(self.data["Education Level"], self.data["Salary"], 1)
        axs[1].plot(self.data["Education Level"], m2 * self.data["Education Level"] + b2, color="red",
                    label="regression line")
        axs[1].set_title('Education Level vs Salary')
        axs[1].legend()
        axs[1].grid()
        axs[2].scatter(self.data["Years of Experience"], self.data["Salary"], color="green", label="datapoints")
        m3, b3 = np.polyfit(self.data["Years of Experience"], self.data["Salary"], 1)
        axs[2].plot(self.data["Years of Experience"], m3 * self.data["Years of Experience"] + b3, color="red",
                    label="regression line")
        axs[2].legend()
        axs[2].grid()
        axs[2].set_title('Years of Experience vs Salary')
        """)
        st.code(code, language="python")
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # legends
        plt.legend()
        # Show the plot
        st.pyplot(fig)
        st.markdown("---")
        st.subheader("Scatter Plot Analysis")
        st.markdown("---")
        st.text("""The scatter plot aims at creating simple regression lines\n
        Equation = y = mx + b\n 
        m = coefficient/weight\n
        b = intercept\n
        y = dependent variable\n
        x = independent variable\nFrom the plots:\n
        -> linearly independent variables - Age / Years of Experience - Cluster them and then use Regression model\n
        -> non-linearly independent variables - Education Level - Polynomial Regression\n

        """)

    def anomalyDetection(self):
        copy = self.data
        st.markdown("---")
        st.subheader("Anomaly Detection")
        st.markdown("---")
        st.text("""There are multiple outlier removal techniques\n
        1.Inter-Quartile-Range(IQR) methods - Use of box plots to remove outliers\n 
            -->disadvantage - makes assumptions that the data is symmetrical\n
        2.Isolation Forest - This is a machine learning algorithm that detects anomalies\n 
            -->advantage - does not make assumptions that data is symmetrical since some extreme values may have significance\nIn this project we will use isolation forest due to its robustness        
        """)
        st.markdown("---")
        st.subheader("Anomaly Detection Models Hyper-Parameter-Tuning")
        st.markdown("---")
        # Drop Outliers
        code = """
        # importation of modules
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.neighbors import LocalOutlierFactor
        # Launching the models
        model_params = {
            "IsolationForest": IsolationForest(contamination=0.13),
            "LocalFactorOutlier": LocalOutlierFactor(contamination=0.13, n_neighbors=10),
            "OneClassSVM": OneClassSVM(nu=0.13)
        }
        # fitting and predicting the models
        for model_name, model in model_params.items():
            self.data[model_name] = model.fit_predict(self.data[["Age", "Education Level", "Years of Experience"]])

        # creation of the y_true column    
        def get_y_true(row):
            mode = row[["LocalFactorOutlier", "OneClassSVM"]].mode()
            if len(mode) == 1:
                return mode
            else:
                return row[["LocalFactorOutlier", "OneClassSVM"]].mode()[0]

        self.data["y_true"] = self.data.apply(get_y_true, axis=1)
        self.data = self.data.drop(columns = ["LocalFactorOutlier", "OneClassSVM"], axis = 1)
        # calculating the f1 score
        from sklearn.metrics import f1_score
        y_pred = self.data["IsolationForest"]
        y_true = self.data["y_true"]
        score = f1_score(y_true, y_pred)
        print(score)
        self.data = self.data[self.data["IsolationForest"] == 1]
        self.data = self.data.drop(columns = ["IsolationForest", "y_true"], axis = 1)
        """
        st.code(code, language="python")
        st.text("""
        The code above simply does the following steps:\n
        1. Import Three anomaly detection models:\n
                 -->We use the three models inorder to come up with the most appropriate y_true column for getting the f1_score:\n
                          f1_score = 2 * precision * recall / precision +  recall\n 
                 -->This is because we do not want to be biased in our anomaly detection\n 
                 -->Models implemented:\n
                     1.LocalFactorOutlier\n
                     2.OnceClassSVM\n
                     3.Isolation Forest\n     
        2. Create the predicted anomaly column for the three models where(pos_labels):\n
                 -1 represents an anomaly\n 
                 +1 represents an non-anomaly\n                 
        3. Now create the y_true column based on the mode of the two helping models - If the two have different values employ Isolation Forest\n
        4. Drop the model predicted anomaly label columns of the two assisting models\n
        5. Evaluate IsolationForest labels against the true column obtained using f1_score\n 
        6. Drop the anomaly rows\n        
        7. Drop the y_true column and IsolationForest column\n
        """)
        st.markdown("---")
        st.subheader("Output of the code above")
        st.markdown("---")
        st.text("Summary Info before anomaly removal")
        buffer2 = StringIO()
        copy.info(buf=buffer2)
        st.text(buffer2.getvalue())
        st.text("Initial Dataset")
        st.write(copy)
        st.text("After Anomaly model predictions")
        # importation of modules
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.neighbors import LocalOutlierFactor
        # Launching the models

        model_params = {
            "IsolationForest": IsolationForest(contamination=0.13),
            "LocalFactorOutlier": LocalOutlierFactor(contamination=0.13, n_neighbors=10),
            "OneClassSVM": OneClassSVM(nu=0.13)
        }
        # fitting and predicting the models
        for model_name, model in model_params.items():
            copy[model_name] = model.fit_predict(copy[["Age", "Education Level", "Years of Experience"]])
        st.write(copy)
        st.text("Generating of the y_true column")

        def get_y_true(row):
            mode = row[["LocalFactorOutlier", "OneClassSVM"]].mode()
            if len(mode) == 1:
                return mode
            else:
                return row[["LocalFactorOutlier", "OneClassSVM"]].mode()[0]

        copy["y_true"] = copy.apply(get_y_true, axis=1)
        st.write(copy)
        st.text("Dropping of the Assisting isolation model columns")
        copy = copy.drop(columns=["LocalFactorOutlier", "OneClassSVM"], axis=1)
        st.write(copy)
        st.text("output of the score")
        from sklearn.metrics import f1_score
        y_pred = copy["IsolationForest"]
        y_true = copy["y_true"]
        score = f1_score(y_true, y_pred)
        st.write(score)
        st.text("Dropping the anomaly rows")
        copy = copy[copy["IsolationForest"] == 1]
        st.write(copy)
        st.text("Dropping the IsolationForest and y_true label columns")
        copy = copy.drop(columns=["IsolationForest", "y_true"], axis=1)
        st.write(copy)
        st.text("Summary info after anomaly removal")
        buffer1 = StringIO()
        copy.info(buf=buffer1)
        st.text(buffer1.getvalue())
        st.markdown("---")
        st.subheader("Final Remark")
        st.markdown("---")
        st.text("""
        From the above procedures our model is ready for modelling:\n
            1.Data is labelled\n
            2.Data is scaled\n
            3.Data is normalized\n
            4.Anomalies have been removed - we can see from the f1_score that our data is highly preprocessed\n
        """)

    # This will return the data after anomaly detection
    def finalResult(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.neighbors import LocalOutlierFactor
        # Launching the models

        model_params = {
            "IsolationForest": IsolationForest(contamination=0.13),
            "LocalFactorOutlier": LocalOutlierFactor(contamination=0.13, n_neighbors=10),
            "OneClassSVM": OneClassSVM(nu=0.13)
        }
        # fitting and predicting the models
        for model_name, model in model_params.items():
            self.data[model_name] = model.fit_predict(self.data[["Age", "Education Level", "Years of Experience"]])

        # creation of the y_true column
        def get_y_true(row):
            mode = row[["LocalFactorOutlier", "OneClassSVM"]].mode()
            if len(mode) == 1:
                return mode
            else:
                return row[["LocalFactorOutlier", "OneClassSVM"]].mode()[0]

        self.data["y_true"] = self.data.apply(get_y_true, axis=1)
        self.data = self.data.drop(columns=["LocalFactorOutlier", "OneClassSVM"], axis=1)
        # calculating the f1 score
        from sklearn.metrics import f1_score
        y_pred = self.data["IsolationForest"]
        y_true = self.data["y_true"]
        score = f1_score(y_true, y_pred)
        self.data = self.data[self.data["IsolationForest"] == 1]
        self.data = self.data.drop(columns=["IsolationForest", "y_true"], axis=1)
        return self.data
