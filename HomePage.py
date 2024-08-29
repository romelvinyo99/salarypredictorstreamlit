import streamlit as st

from Summaries import Summaries


class Home:
    def __init__(self, df):
        self.data = df
        st.header("Dataset Report")
        st.write("Salary Dataset")
        st.write(df)
        st.subheader("Overview")
        st.write("""
        The dataset contains 6 columns with the following names:\n
        \t1.Age\n
        \t2.Gender\n
        \t3.Education Level\n
        \t4.Job Title\n
        \t5.Years of Experience\n
        \t6.Salary\n
            """)
        st.subheader("Summary Information")
        info = Summaries(self.data)
        info.SummaryInfo()


