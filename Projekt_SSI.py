import streamlit as st
import pickle as pk
import numpy as np

st.title("Projekt SSI")
with st.form(key='data_form'):
    st.text('Wypełnij formularz i sprawdź czy byś przeżył!')
    sex = st.radio('Płeć:',['M', 'K'])
    cls = st.selectbox('Klasa: ',['1','2','3'])
    sibsp = st.number_input('Liczba rodzeństwa + małżonek na pokładzie:', 0, 10)
    parch = st.number_input('Liczba rodziców/dzieci na pokładzie:', 0, 10)
    fare = st.number_input('Opłata za bilet:', 0.0, 512.0)
    embarked = st.radio('Port startowy pasażera:', ['Cherbourg', 'QuinsTown','Southampton'])
    submit = st.form_submit_button('Sprawdź!')

if submit:
    data = np.array([[int(cls), sex=='M', sibsp, parch, fare, embarked=='Cherbourg', embarked=='QuinsTown', embarked=='Southampton']])
    with open('decision_tree.pickle','rb') as f:
        dt = pk.load(f)
    surv = dt.predict(data)
    if surv[0] == 1:
        st.success('Przeżyłeś!')
        st.balloons()
    else:
        st.error('Nie przeżyłeś!')