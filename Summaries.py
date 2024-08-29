import streamlit as st
from io import StringIO


class Summaries:
    def __init__(self, df):
        self.data = df

    def SummaryInfo(self):
        buffer = StringIO()
        self.data.info(buf=buffer)
        st.text(buffer.getvalue())




