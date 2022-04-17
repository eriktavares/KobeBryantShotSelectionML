import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
############################################ SIDE BAR TITLE

#st.sidebar.title('Kobe Bryant Shot Selection ML')

#st.selectbox("Select Dataset",)

st.sidebar.title("Menu")
pages=["Inicial","Registro","Operação", "Diagrama ML"]
paginaselecionada=st.sidebar.selectbox("Opções",pages)




#Log Experimento
experiment_name = 'Kobe_Bryant_Shot_Experiment'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id


def inicial():
    st.title('Kobe Bryant Shot Selection ML')
    st.image("../Docs/Data_Report/Images/kb_presentation.gif")

def tracking():
    pass


def registro_logloss():
    df_ml = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rg = df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RegistroModelo'].copy()
    df_ml_fh_rg = df_ml_fh_rg[df_ml_fh_rg['metrics.LogLoss'].notnull()].reset_index()
    df_ml_fh_rg = df_ml_fh_rg[df_ml_fh_rg['metrics.Version'].notnull()].reset_index()

    df_ml = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rv= df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RevalidaçãoModelo'].copy()
    df_ml_fh_rv = df_ml_fh_rv[df_ml_fh_rv['metrics.LogLoss'].notnull()].reset_index()
    df_ml_fh_rv= df_ml_fh_rv[df_ml_fh_rv['metrics.Version'].notnull()].reset_index()




    fig, ax = plt.subplots(sharex=True)
    ax.set_title('Medica Log Loss')
    plt.xlabel("Versão", fontsize=30)
    plt.ylabel("Log Loss", fontsize=30)
    plt.plot(df_ml_fh_rg['metrics.Version'], df_ml_fh_rg['metrics.LogLoss'], 'o-', label="Log Loss", linewidth=4)
    plt.plot(df_ml_fh_rg['metrics.Version'], df_ml_fh_rg['metrics.LogLoss'], 'o-', label="Log Loss", linewidth=4)
    plt.legend()
    st.pyplot(fig)


def registro():
    opcoes2 = ["LogLoss"]
    paginaselecionada2 = st.sidebar.selectbox("Visualização", opcoes2)
    if(paginaselecionada2==opcoes2[0]):
        registro_logloss()



def operalization():
    dados_resultados = pd.read_parquet('../Data/Operalization/results/results01.parquet')
    dados_resultados




if(paginaselecionada==pages[0]):
    inicial()

elif((paginaselecionada==pages[1])):
    registro()

elif((paginaselecionada==pages[2])):
    operalization()

#Diagrama ML
elif(paginaselecionada==pages[3]):
    st.image("https://docs.google.com/drawings/d/1jJVdbzMk1Bs8MjJGsUxDY2DpjmaMfJQvlegU4KWrdso/edit?usp=sharing")

def dadosa():
    pass


def diagrama_ml():
    st.title('Diagrama ML')

