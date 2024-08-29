import streamlit as st
from io import StringIO
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, RandomizedSearchCV, train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import r2_score


class HyperParameterTuning:
    def __init__(self, df):
        self.data = df
        self.copy = self.data
        km = KMeans(n_clusters=3)
        self.copy["clusters"] = km.fit_predict(self.copy[["Education Level"]])

    def KMeans(self):
        st.header("KMeans")
        st.subheader("Arm Technique")
        st.write("""
        --> This technique will help us get the best value for clusters\n
        --> The best value is the value where the curve becomes flat
        """)
        st.subheader("Code")
        st.code("""
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        import numpy as np
        SSE = []
        k_range = range(1, 11)
        for k in k_range:
            km = KMeans(n_clusters = k)
            km.fit_predict(df[["Education Level"]])
            SSE.append(km.inertia_)    
        plt.plot(k_range, SSE, label = "arm plot")
        plt.xlabel("k")
        plt.ylabel("SSE")
        plt.legend()
        """, language="python")
        st.subheader("result")
        SSE = []
        k_range = range(1, 11)
        for k in k_range:
            km = KMeans(n_clusters=k)
            km.fit_predict(self.data[["Education Level"]])
            SSE.append(km.inertia_)
        fig, axs = plt.subplots(figsize=(12, 6))
        axs.plot(k_range, SSE, label="arm plot")
        axs.set_xlabel("k")
        axs.set_ylabel("SSE")
        axs.legend()
        st.pyplot(fig)
        st.subheader("Clustering")
        st.write("We will create a new column representing the clusters")
        km = KMeans(n_clusters=3)
        self.data["clusters"] = km.fit_predict(self.data[["Education Level"]])
        st.code("""
        km = KMeans(n_clusters = 3)
        df["clusters"] = km.fit_predict(non_linear_df[["Education Level"]
        df.nunique()
        print(df.head(3))
        """, language="python")
        st.write("result")
        st.write(self.data)
        st.subheader("Education level clusters plot")
        st.code("""
        cluster1 = df[df["clusters"] == 0]
        cluster2 = df[df["clusters"] == 1]
        cluster3 = df[df["clusters"] == 2]
        plt.scatter(cluster1["Education Level"], cluster1["Salary"], label = "cluster1")
        plt.scatter(cluster2["Education Level"], cluster2["Salary"], label = "cluster2")
        plt.scatter(cluster3["Education Level"], cluster3["Salary"], label = "cluster3")
        plt.scatter(km.cluster_centers_,km.cluster_centers_, marker = "+",  color = "purple", label = "centroids")
        plt.ylabel("salary")
        plt.xlabel("education level")
        plt.xticks(ticks = [0, 0.5, 1], labels = ["bachelor's", "masters", "phd"])
        print(f"cluster centers = {km.cluster_centers_}")
        """)
        cluster1 = self.data[self.data["clusters"] == 0]
        cluster2 = self.data[self.data["clusters"] == 1]
        cluster3 = self.data[self.data["clusters"] == 2]
        fig, axs = plt.subplots(figsize=(12, 6))
        axs.scatter(cluster1["Education Level"], cluster1["Salary"], label="cluster1")
        axs.scatter(cluster2["Education Level"], cluster2["Salary"], label="cluster2")
        axs.scatter(cluster3["Education Level"], cluster3["Salary"], label="cluster3")
        axs.scatter(km.cluster_centers_, km.cluster_centers_, marker="+", color="purple", label="centroids")
        axs.set_ylabel("salary")
        axs.set_xlabel("education level")
        axs.legend()
        axs.set_xticks(ticks=[0, 0.5, 1], labels=["bachelor's", "masters", "phd"])
        st.pyplot(fig)
        print(f"cluster centers = {km.cluster_centers_}")

    def linearRegressionModels(self):
        st.header("Statistical Hyper Parameter Tuning(Linear and Non-Linear data)")
        st.markdown("---")
        st.subheader("1.Overview")
        st.markdown("---")
        st.write(self.data)
        st.write("""
               -->From the data analysis we drew the following inferences:\n
                   1. X = Age - Education Level - Years of experience\n
                   2. y = Salary\n
               -->This suggest that if we are building a model we are building a regression model\n
                   1. Age - Years of Experience - Datapoints are linearly independent\n    
                        -->This means we will model this features using linear regression\n
                               y = mx + b --> base equation\n
                   2. Education Level - Data points are not linearly independent\n
                        -->This means we will first cluster the datapoints and then model incorporate the clusters into our model build
               -->We will start by hyper parameter tunes for each class of problems to get the best parameters\n
               -->This will be achieved by using Grid Search Algorithm\n
                        nCR = n! / (n-r)! x r! --> n choose r --> combinations based on the parameters\n    
               -->I have reduced the parameters due to insufficient computational power - It was run in the background recursively                        
               """)
        self.KMeans()
        st.markdown("---")
        st.subheader("1.Linear Regression Hyper-Parameter Tunes")
        st.markdown("---")
        code = """
        from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV, KFold

        X = df.drop(columns=["Salary", "Education Level"])
        y = df["Salary"]

        model_params1 = {
            "LinearRegression": {
                "model": LinearRegression(),
                "params": {}
            },
            "LassoRegression": {
                "model": Lasso(),
                "params": {
                    "alpha": [0.482, 0.481, 0.483],
                    "tol": [0.0616, 0.0612, 0.061],
                    "selection": ["cyclic", "random"],
                    "warm_start": [False, True],
                    "precompute": [False, True],
                    "fit_intercept": [True, False]
                }
            },
            "RidgeRegression": {
                "model": Ridge(),
                "params": {
                    "alpha": [1.03028, 1.03026, 1.03025],
                    "tol": [0.003, 0.031, 0.032],
                    "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                    "copy_X": [True, False],
                    "fit_intercept": [True, False]
                }
            }
        }

        score1 = []
        kf = KFold(n_splits=3)

        for model_name, mp in model_params1.items():
            gs1 = GridSearchCV(mp["model"], mp["params"], cv=5, return_train_score=False)
            gs1.fit(X, y)
            score1.append({
                "model": model_name,
                "Best Score": gs1.best_score_,
                "Best Parameter": gs1.best_params_
            })

        print(score1)
        """

        st.code(code, language="python")
        st.code("""
        from sklearn.ensemble import RandomForestRegressor
        model_params2 = {
            "RandomForestRegressor": {
                "model": RandomForestRegressor(warm_start=True, bootstrap=True, n_estimators=50,
                                               criterion="absolute_error"),
                "params": {}
            }
        }
        score2 = []
        for model_name1, mp1 in model_params2.items():
            gs = GridSearchCV(mp1["model"], mp1["params"], return_train_score=False)
            gs.fit(X, y)
            score2.append({"model": model_name1, "best score": [gs.best_score_], "best parameters": gs.best_params_})
        """)

        st.subheader("Output")
        st.write(self.data)
        X = self.data.drop(columns=["Salary", "Education Level"])
        y = self.data["Salary"]

        model_params1 = {
            "LinearRegression": {
                "model": LinearRegression(),
                "params": {}
            },
            "LassoRegression": {
                "model": Lasso(),
                "params": {
                    "alpha": [0.482, 0.481, 0.483],
                    "tol": [0.0616, 0.0612, 0.061],
                    "selection": ["cyclic", "random"],
                    "warm_start": [False, True],
                    "precompute": [False, True],
                    "fit_intercept": [True, False]
                }
            },
            "RidgeRegression": {
                "model": Ridge(),
                "params": {
                    "alpha": [1.03028, 1.03026, 1.03025],
                    "tol": [0.003, 0.031, 0.032],
                    "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                    "copy_X": [True, False],
                    "fit_intercept": [True, False]
                }
            }
        }

        score1 = []
        for model_name, mp in model_params1.items():
            gs1 = GridSearchCV(mp["model"], mp["params"], cv=5, return_train_score=False)
            gs1.fit(X, y)
            score1.append({
                "model": model_name,
                "Best Score": gs1.best_score_,
                "Best Parameter": gs1.best_params_
            })
        from sklearn.ensemble import RandomForestRegressor
        model_params2 = {
            "RandomForestRegressor": {
                "model": RandomForestRegressor(warm_start=True, bootstrap=True, n_estimators=50,
                                               criterion="absolute_error"),
                "params": {}
            }
        }
        score2 = []
        for model_name1, mp1 in model_params2.items():
            gs = GridSearchCV(mp1["model"], mp1["params"], return_train_score=False)
            gs.fit(X, y)
            score2.append({"model": model_name1, "best score": [gs.best_score_], "best parameters": gs.best_params_})

        st.write(score1)
        st.write(score2)
        st.subheader("Explanation")
        st.text("""
        The code above simply performs a grid search on the models and the parameter limit set\n
        The result shows us the parameters that give us the best score\n
        For the SGD regressor the computation time is too much so hyper-parameter tuning was done recursively\n
                 1.max_iter=1000000\n
                 2.loss="squared_epsilon_insensitive"\n
                 3.penalty="l2"\n
        The rest are the default model parameters\n         
        """)
        st.subheader("Cross validation")
        st.code("""
        s1, s2, s3, s4, s5 = cross_val_score(Lasso(alpha = 0.481, fit_intercept=True, precompute=True, selection="random", tol=0.0612, warm_start=True),X, y,  cv=5)
        average1 = (s1 + s2 + s3 + s4 + s5) / 5
        s1, s2, s3, s4, s5 = cross_val_score(LinearRegression(), X, y, cv = 5)
        average2 = (s1 + s2 + s3 + s4 + s5) / 5
        s1, s2, s3, s4, s5 = cross_val_score(Ridge(alpha =  1.03028,copy_X= False,fit_intercept= True,solver= 'saga',tol= 0.031), X, y, cv=5)
        average3 = (s1 + s2 + s3 + s4 + s5) / 5
        s1, s2, s3, s4, s5 = cross_val_score(SGDRegressor(max_iter=1000000, loss="squared_epsilon_insensitive", penalty="l2"),X, y, cv=5)
        average4 = (s1+s2+s3+s4+s5) / 5
        print(f"average{i} for i in range(1, 5)")
        """)
        s1, s2, s3, s4, s5 = cross_val_score(
            Lasso(alpha=0.482, fit_intercept=True, precompute=True, selection="random", tol=0.061, warm_start=True), X,
            y, cv=5)
        average1 = (s1 + s2 + s3 + s4 + s5) / 5
        s1, s2, s3, s4, s5 = cross_val_score(LinearRegression(), X, y, cv=5)
        average2 = (s1 + s2 + s3 + s4 + s5) / 5
        s1, s2, s3, s4, s5 = cross_val_score(
            Ridge(alpha=1.03025, copy_X=False, fit_intercept=True, solver='sag', tol=0.032), X, y, cv=5)
        average3 = (s1 + s2 + s3 + s4 + s5) / 5
        s1, s2, s3, s4, s5 = cross_val_score(
            SGDRegressor(max_iter=1000000, loss="squared_epsilon_insensitive", penalty="l2"), X, y, cv=5)
        average4 = (s1 + s2 + s3 + s4 + s5) / 5
        st.write(f"""
        Average score for Lasso = {average1}\n
        Average score for Linear Regression = {average2}\n
        Average score for Ridge = {average3}\n
        Average score for SGD Regressor = {average4}\n
        """)

    def backdoor(self):
        st.markdown("---")
        st.subheader("Linear Regression Background - Gradient Descent")
        st.markdown("---")
        copy = self.data

        st.code('''
        cost_list = []
        m1_list = []
        m2_list = []
        bias_list = []

        def gradient_descent(X1, X2, y, iterations=10000000, learning_rate=0.1, loss_threshold=0.10):
            m1 = m2 = 1
            bias = 0
            n = len(X1)
            X1 = np.array(X1)
            X2 = np.array(X2)
            y = np.array(y)

            for i in range(iterations):
                y_predicted = m1 * X1 + m2 * X2 + bias
                cost = np.mean((y - y_predicted) ** 2)
                m1d = (-2 / n) * np.sum(X1 * (y - y_predicted))
                m2d = (-2 / n) * np.sum(X2 * (y - y_predicted))
                bd = (-2 / n) * np.sum(y - y_predicted)

                m1 -= learning_rate * m1d
                m2 -= learning_rate * m2d
                bias -= learning_rate * bd

                # Append values to lists
                m1_list.append(m1)
                m2_list.append(m2)
                bias_list.append(bias)
                cost_list.append(cost)

                print(f"Iteration {i}: m1 = {m1}, m2 = {m2}, bias = {bias}, cost = {cost}")

                # Check for convergence
                if cost < loss_threshold:
                    print(f"Convergence reached at iteration {i}")
                    break

            return m1, m2, bias

        # Example usage
        gradient_descent(df["Age"], df["Years of Experience"], df["Salary"])
        ''', language='python')
        cost_list = []
        m1_list = []
        m2_list = []
        bias_list = []

        def gradient_descent(X1, X2, y, iteration=100, learning_rate=0.1):
            m1 = m2 = 1
            bias = 0
            n = len(X1)
            X1 = np.array(X1)
            X2 = np.array(X2)
            y = np.array(y)
            for i in range(iteration):
                y_predicted = m1 * X1 + m2 * X2 + bias
                cost = np.mean((y - y_predicted) ** 2)
                m1d = (-2 / n) * np.sum(X1 * (y - y_predicted))
                m2d = (-2 / n) * np.sum(X2 * (y - y_predicted))
                bd = (-2 / n) * sum((y - y_predicted))
                m1 = m1 - learning_rate * m1d
                m2 = m2 - learning_rate * m2d
                bias = bias - learning_rate * bd
                m1_list.append(m1.tolist())
                m2_list.append(m2.tolist())
                bias_list.append(bias.tolist())
                cost_list.append(cost.tolist())
                print(f"iteration = {i} m1_current = {m1} m2_current = {m2} bias_current = {bias} cost = {cost}")
            return m1, m2, bias

        st.subheader("Result")
        gradient_descent(self.data["Age"], self.data["Years of Experience"], self.data["Salary"])
        cost_df = pd.DataFrame({
            "cost": cost_list,
            "m1": m1_list,
            "m2": m2_list,
            "bias": bias_list
        })
        cost_df = cost_df.dropna()
        st.subheader("Cost function plots")
        fig, axs = plt.subplots(1, 3, figsize=(15, 15))
        axs[0].plot(cost_df["m1"], cost_df["cost"])
        axs[0].set_xlabel("m1")
        axs[0].set_ylabel("cost")
        axs[1].plot(cost_df["m2"], cost_df["cost"])
        axs[1].set_xlabel("m2")
        axs[1].set_ylabel("cost")
        axs[2].plot(cost_df["bias"], cost_df["cost"])
        axs[2].set_xlabel("bias")
        axs[2].set_ylabel("cost")
        st.pyplot(fig)
        st.subheader("Explanation")
        st.text("""
        Gradient descent tends to find the best coefficient and bias which has the lowest cost\n
        In our regression problem the cost is mean squared error:\n
                     mse = 1/n * sum of (y - y predicted) squared\n
        Base case:\n
                     y = m * x + bias\n
                     m = coefficient\n
                     bias = intercept\n
                     iteration = number of adjustments\n
                     learning_rate = size of steps with each adjustment\n
        Tuning:\n
                --> Find the derivative of mse with respect to m and bias\n 
                     m derivative = (-2/n) * sum (x * (y - y_predicted))\n
                     bias derivative = (-2/n) * sum(y - y_predicted)\n
                --> Readjust the values of the m and bias with each iteration\n
                    m_new = m_current - learning_rate * m derivative\n
                    bias_new = b_current - learning_rate * bias derivative\n           
        Global minimum - The main aim of gradient descent is for the curves to start from a high point and then approach a global minimum\n 
        From the global minimum we extract the coefficient and intercept values\n                                           
        """)

    def final(self):
        return self.data


class StatsModelling:
    def __init__(self, clustered_data):
        self.data = clustered_data

    @staticmethod
    def userScaling(scaler, age, level, experience):
        age, level, experience = scaler.transform([[age, level, experience]]).flatten()
        return age, level, experience

    def fitting(self, estimator):
        X = self.data.drop(columns=["Education Level", "Salary"])
        y = self.data["Salary"]
        X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2)
        kf = KFold(n_splits=10)
        score = []
        score_prev = 0
        score_next = 0
        maxi_score = 0
        best_estimator = LinearRegression()
        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
                test_index]
            estimator.fit(X_train, y_train)
            score_next = estimator.score(X_test, y_test)
            if score_prev > score_next:
                maxi_score = score_prev
            else:
                maxi_score = score_next
                best_estimator = estimator
            score.append(estimator.score(X_test, y_test))
            score_prev = estimator.score(X_test, y_test)
        average_score = np.mean(score)
        best_estimator.fit(X1, y1)
        best_score = best_estimator.score(X2, y2)
        return best_estimator, best_score, average_score

    def linearRegression(self):
        best_estimator, best_score, average_score = self.fitting(LinearRegression())
        df = pd.DataFrame({
            "Metric": ["Best Estimator Score", "Average Accuracy Score", "estimator"],
            "Value": [best_score, average_score, best_estimator]
        })
        st.write(df)
        return best_estimator

    def lassoRegression(self):
        best_estimator, best_score, average_score = self.fitting(
            Lasso(alpha=0.482, fit_intercept=True, precompute=False, selection="random", tol=0.0616, warm_start=False))
        df = pd.DataFrame({
            "Metric": ["Best Estimator Score", "Average Accuracy Score", "estimator"],
            "Value": [best_score, average_score, best_estimator]
        })
        st.write(df)
        return best_estimator

    def ridgeRegression(self):
        best_estimator, best_score, average_score = self.fitting(
            Ridge(alpha=1.03028, fit_intercept=True, solver="saga", tol=0.032, copy_X=True))
        df = pd.DataFrame({
            "Metric": ["Best Estimator Score", "Average Accuracy Score", "estimator"],
            "Value": [best_score, average_score, best_estimator]
        })
        st.write(df)
        return best_estimator

    def randomForestRegression(self):
        best_estimator, best_score, average_score = self.fitting(
            RandomForestRegressor(warm_start=True, bootstrap=True, n_estimators=50, criterion="absolute_error"))
        df = pd.DataFrame({
            "Metric": ["Best Estimator Score", "Average Accuracy Score", "estimator"],
            "Value": [best_score, average_score, best_estimator]
        })
        st.write(df)
        return best_estimator

    @staticmethod
    def predict(age, level, experience, estimator):
        salary = estimator.predict([[age, level, experience]])
        return salary

    @staticmethod
    def inputField():
        st.markdown("---")
        st.subheader("Prediction")
        st.markdown("---")
        age = int(st.text_input("Input Age: ", value=0))
        default_text = "Education Level"
        level = st.selectbox("Education", ["Bachelor's", "Master's", "PhD"])
        experience = int(st.text_input("Input Years of Experience: ", value=0))
        if level == "Bachelor's":
            level = 0
        elif level == "Master's":
            level = 1
        elif level == "PhD":
            level = 2
        else:
            raise Exception("Invalid input")
        return age, level, experience


class DeepLearning:
    def __init__(self, clustered_data):
        self.data = clustered_data

    def model(self):
        X = self.data.drop(columns=["Salary", "Education Level"], axis=1)
        y = self.data["Salary"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(25, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear")
        ])
        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["mean_absolute_error"]
        )
        model.fit(X_train, y_train, validation_split=0.15, epochs=1100)
        y_predicted = model.predict(X_test)
        score = r2_score(y_test, y_predicted)
        st.success(f"The model accuracy Score = {score * 100}%")
        return model

    @staticmethod
    def predict(age, level, experience, estimator):
        X_new = pd.DataFrame({
            "age": [age],
            "experience": [experience],
            "level": [level]
        })
        salary = estimator.predict(X_new)
        return salary
