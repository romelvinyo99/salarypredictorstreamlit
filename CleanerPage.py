import pandas as pd
import streamlit as st
from Summaries import Summaries


class Cleaner:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)

    def dropNullValues(self):
        st.header("Data Cleaning")
        st.markdown("---")
        st.write("""The following operations will be performed:\n
                        \t1.Checking for null values\n\n\t2.Checking for duplicates\n""")
        info = Summaries(self.data)
        info.SummaryInfo()
        st.markdown("---")

        st.subheader("1.Null values")
        st.markdown("---")
        code1 = """
            #checking the sum
            print(df.isnull().sum()
            """
        st.code(code1, language="python")
        st.text(f"Total = {self.data.isna().sum().sum()}\n\nThe rows are as follows")
        st.write(self.data[self.data.isnull().any(axis=1)])
        st.write("Dropping the null values")
        code2 = """
            df = df.dropna()
            """
        st.code(code2, language="python")
        st.markdown("---")

    def dropDuplicates(self):
        copy = self.data
        st.subheader("2.Duplicated values")
        st.markdown("---")
        code3 = """
                #checking the sum
                print(df.duplicated()
                """
        st.code(code3, language="python")
        st.text(f"Total = {self.data.duplicated().sum()}\n\nThe rows are as follows")
        st.write(self.data[self.data.duplicated()])
        st.text("Dropping the duplicated values")
        code4 = """
                df = df.drop_duplicates()
                """
        st.code(code4, language="python")
        copy = copy.dropna().drop_duplicates()
        st.subheader("3.Final result")

        info = Summaries(copy)
        info.SummaryInfo()
        st.markdown("---")
        st.text("The duplicated values and null values dropped")

    def finalResult(self):
        self.data = self.data.dropna().drop_duplicates()
        return self.data


