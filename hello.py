import streamlit as st

st.title("hello world")


# slider
score = st.slider('Please specify your test score',
                  min_value=0, max_value=100, value=10)
st.write("My test score is ", score)