import helper
import streamlit as st 
import pickle

model=pickle.load(open(r"artfacts/lr.pkl",'rb'))


text=st.text_input('enter your review')


text=helper.text_preprocessing(text).toarray()


if st.button("predict") :
    pred=model.predict(text)

st.text(pred)