import streamlit as st
import pandas as pd
############################################ SIDE BAR TITLE

#st.sidebar.title('Kobe Bryant Shot Selection ML')

#st.selectbox("Select Dataset",)

st.sidebar.title("Menu")
pages=["Inicial", "Operação", "Diagrama ML"]
paginaselecionada=st.sidebar.selectbox("Opções",pages)


def inicial():
    st.title('Kobe Bryant Shot Selection ML')
    st.image("../Images/kb_presentation.gif")

def operalization():
    dados_resultados = pd.read_parquet('../Data/Operalization/results/results01.parquet')
    dados_resultados

if(paginaselecionada==pages[0]):
    inicial()

elif((paginaselecionada==pages[1])):
    operalization()

#Diagrama ML
elif(paginaselecionada==pages[2]):
    st.image("https://docs.google.com/drawings/d/1jJVdbzMk1Bs8MjJGsUxDY2DpjmaMfJQvlegU4KWrdso/edit?usp=sharing")

def dadosa():
    pass


def diagrama_ml():
    st.title('Diagrama ML')
