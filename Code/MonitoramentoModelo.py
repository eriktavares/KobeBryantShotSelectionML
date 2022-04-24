import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn import metrics
import numpy as np
############################################ SIDE BAR TITLE

#st.sidebar.title('Kobe Bryant Shot Selection ML')

#st.selectbox("Select Dataset",)
KB_PRESENTATION_GIF = "../Docs/Data_Report/Images/kb_presentation.gif"
operation_processed_parquet_path = '../Data/Operalization/base_operation_processed.parquet'
target_col = 'shot_made_flag'
st.sidebar.title("Menu")
pages=["Inicial","Versionamento","Operação", "Diagrama ML"]
paginaselecionada=st.sidebar.selectbox("Opções",pages)

#st.area_chart(data=None, width=10000, height=5, use_container_width=True)


#Log Experimento
experiment_name = 'Kobe_Bryant_Shot_Experiment'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id


def inicial():
    st.title('Kobe Bryant Shot Selection ML')
    st.image("%s" % KB_PRESENTATION_GIF)

def tracking():
    pass


def registro_logloss():
    df_ml = get_mlflow_experiments()
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rg = df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RegistroModelo'].copy()
    df_ml_fh_rg = df_ml_fh_rg[df_ml_fh_rg['metrics.LogLoss'].notnull()].reset_index()
    df_ml_fh_rg = df_ml_fh_rg[df_ml_fh_rg['metrics.Version'].notnull()].reset_index()

    df_ml = get_mlflow_experiments()
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rv= df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RevalidaçãoModelo'].copy()
    df_ml_fh_rv = df_ml_fh_rv[df_ml_fh_rv['metrics.LogLoss'].notnull()].reset_index()
    df_ml_fh_rv= df_ml_fh_rv[df_ml_fh_rv['metrics.Version'].notnull()].reset_index()

    fig, ax = plt.subplots(sharex=True)
    ax.set_title('Metrica Log Loss')
    plt.xlabel("Versão", fontsize=30)
    plt.ylabel("Log Loss", fontsize=30)
    plt.plot(df_ml_fh_rg['metrics.Version'], df_ml_fh_rg['metrics.LogLoss'], 'o-', label="Registro", linewidth=4)
    plt.plot(df_ml_fh_rv['metrics.Version'], df_ml_fh_rv['metrics.LogLoss'], 'o-', label="Operação", linewidth=4)
    plt.legend()
    st.pyplot(fig)


def get_mlflow_experiments():
    return mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])


def registro_f1():
    df_ml = get_mlflow_experiments()
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rg = df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RegistroModelo'].copy()
    df_ml_fh_rg = df_ml_fh_rg[df_ml_fh_rg['metrics.F1'].notnull()].reset_index()
    df_ml_fh_rg = df_ml_fh_rg[df_ml_fh_rg['metrics.Version'].notnull()].reset_index()

    df_ml = get_mlflow_experiments()
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rv= df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RevalidaçãoModelo'].copy()
    df_ml_fh_rv = df_ml_fh_rv[df_ml_fh_rv['metrics.F1'].notnull()].reset_index()
    df_ml_fh_rv= df_ml_fh_rv[df_ml_fh_rv['metrics.Version'].notnull()].reset_index()

    fig, ax = plt.subplots(sharex=True)
    ax.set_title('Medica Log Loss')
    plt.xlabel("Versão", fontsize=30)
    plt.ylabel("F1-Score", fontsize=30)
    plt.plot(df_ml_fh_rg['metrics.Version'], df_ml_fh_rg['metrics.F1'], 'o-', label="Registro", linewidth=4)
    plt.plot(df_ml_fh_rv['metrics.Version'], df_ml_fh_rv['metrics.F1'], 'o-', label="Operação", linewidth=4)
    plt.legend()
    st.pyplot(fig)

def registro_accuracy():
    df_ml = get_mlflow_experiments()
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rg = df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RegistroModelo'].copy()
    df_ml_fh_rg = df_ml_fh_rg[df_ml_fh_rg['metrics.Accuracy'].notnull()].reset_index()
    df_ml_fh_rg = df_ml_fh_rg[df_ml_fh_rg['metrics.Version'].notnull()].reset_index()

    df_ml = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rv= df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RevalidaçãoModelo'].copy()
    df_ml_fh_rv = df_ml_fh_rv[df_ml_fh_rv['metrics.Accuracy'].notnull()].reset_index()
    df_ml_fh_rv= df_ml_fh_rv[df_ml_fh_rv['metrics.Version'].notnull()].reset_index()

    fig, ax = plt.subplots(sharex=True)
    ax.set_title('Medica Log Loss')
    plt.xlabel("Versão", fontsize=30)
    plt.ylabel("Acurácia", fontsize=30)
    plt.plot(df_ml_fh_rg['metrics.Version'], df_ml_fh_rg['metrics.Accuracy'], 'o-', label="Registro", linewidth=4)
    plt.plot(df_ml_fh_rv['metrics.Version'], df_ml_fh_rv['metrics.Accuracy'], 'o-', label="Operação", linewidth=4)
    plt.legend()
    st.pyplot(fig)


def operacao_roc():

    df_op = pd.read_parquet(operation_processed_parquet_path)
    fpr, tpr, thresholds = metrics.roc_curve(df_op[target_col], df_op['operation_label'])
    # roc_auc = metrics.auc(fpr, tpr)

    # ax.plot(fpr, tpr,
    #        color='g')
    # ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    fig, ax = plt.subplots(sharex=True, figsize=(15, 15))

    ax.plot(fpr,
            tpr,
            color='g',
            lw=2,
            alpha=0.8)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, label="Chance", alpha=0.8)
    plt.xlabel("False Positive Rate", fontsize=30)
    plt.ylabel("True Positive Rate", fontsize=30)
    #ax.legend(loc="center right", prop={'size': 23})
    st.pyplot(fig)


def operation_accuracy():
    df_ml = get_mlflow_experiments()
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rv= df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RevalidaçãoModelo'].copy()
    df_ml_fh_rv = df_ml_fh_rv.sort_values(['metrics.Version'])
    df_ml_fh_rv = df_ml_fh_rv[df_ml_fh_rv['metrics.Accuracy'].notnull()].reset_index()
    df_ml_fh_rv= df_ml_fh_rv[df_ml_fh_rv['metrics.Version'].notnull()].reset_index()
    df_ml_fh_rv = df_ml_fh_rv.sort_values(['metrics.Version'])
    fig, ax = plt.subplots(sharex=True)
    ax.set_title('Medica Acurácia')
    plt.xlabel("Indice", fontsize=30)
    plt.ylabel("Acurácia", fontsize=30)
    x=[]
    y=[]
    for i in df_ml_fh_rv.index:
        x.append(i)
        y.append(df_ml_fh_rv['metrics.Accuracy'][i])
    plt.plot(x,y , 'o-', label="Registro", linewidth=4)
    plt.legend()
    st.pyplot(fig)

def opertion_acuracia_recall_prec():
    #import matplotlib.pyplot as plt
    #import numpy as np
    df_ml_fh_rv=operacion_metrics();
    df_ml_fh_rv = df_ml_fh_rv.sort_values(['metrics.Version'])
    plt.rcParams["figure.figsize"] = [15, 15]
    plt.rcParams["figure.autolayout"] = True
    labels = df_ml_fh_rv['metrics.Version']
    tipo1 = df_ml_fh_rv['metrics.Prec.']
    tipo2 = df_ml_fh_rv['metrics.Accuracy']
    tipo3 = df_ml_fh_rv['metrics.Recall']
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, tipo1, width, label='Precisão')
    rects2 = ax.bar(x, tipo2, width, label='Acurácia')
    rects3 = ax.bar(x + width, tipo3, width, label='Recall')
    plt.xticks(fontsize=15)
    ax.set_ylabel('Metricas')
    ax.set_ylabel('Versões')
    ax.set_title('Metricas Por Versões')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.xticks(fontsize=15)

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    st.pyplot(fig)

def alarm(text):
    st.warning(text)

def operation_error_acertos_range(valor, valor_min, valor_max, texto_min, texto_max):
    if(valor<valor_min):
        alarm(texto_min)
    if(valor>valor_max):
        alarm(texto_max)




def operation_error_acertos():
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_axes([0, 0, 2, 1])
    df_op = pd.read_parquet(operation_processed_parquet_path)
    types = ['Acertou Ope.', 'Errou Opr.', 'Acertou Real', 'Errou Real']
    percent = [df_op['operation_label'].value_counts(normalize=True)[1],
               df_op['operation_label'].value_counts(normalize=True)[0]]
    percent2 = [df_op['shot_made_flag'].value_counts(normalize=True)[1],
                df_op['shot_made_flag'].value_counts(normalize=True)[0]]
    percent.extend(percent2)
    ax.bar(types, percent)
    # ax.bar(types,percent2)
    ax.set_ylabel('Lances', fontsize=15)
    ax.set_xlabel('Resultado', fontsize=15)
    for index, value in enumerate(percent):
        txt = f'{round(value * 100, 2)} %'
        y_coord = value + 0.005
        x_coord = index - 0.1
        ax.text(x=x_coord, y=y_coord, s=txt, fontsize=15)
        ax.grid()
    #st.area_chart(data=None, width=500, height=500, use_container_width=True)
    st.pyplot(fig)
    form = st.sidebar.form("input_form")
    form.subheader("Proporção Operação")
    percent_input_oper_acerto = form.slider("Acerto",
                                  value=[0,100])
    form.info('Min {}% - Max {}%'.format(percent_input_oper_acerto[0],percent_input_oper_acerto[1]))
    percent_input_oper_erro = form.slider("Erro",
                                  value=[0,100])
    form.info('Min {}% - Max {}%'.format(percent_input_oper_erro [0], percent_input_oper_erro [1]))
    form.form_submit_button()
    alarm_template="Alarme! Percentual de {} {} {:0.2f}% do valor {} {:0.2f}%"
    operation_error_acertos_range(percent[0]*100,
                                  percent_input_oper_acerto[0],
                                  percent_input_oper_acerto[1],
                                  alarm_template.format('Acertos Preditos','Abaixo',percent[0]*100,'Mínimo',percent_input_oper_acerto[0]),
                                  alarm_template.format('Acertos Preditos','Acima',percent[0]*100,'Máximo',percent_input_oper_acerto[1]))
    operation_error_acertos_range(percent[1]*100,
                                  percent_input_oper_erro[0],
                                  percent_input_oper_erro[1],
                                  alarm_template.format('Erros Preditos','Abaixo',percent[1]*100, 'Mínimo',percent_input_oper_erro[0]),
                                  alarm_template.format('Erros Preditos','Acima',percent[1]*100,'Máximo',percent_input_oper_erro[1]))
    #(valor, valor_min, valor_max, texto_min, texto_max)
    #alarm('Alarme')

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:0.2f}%'.format(height * 100),
                    xy=(rect.get_x()+0.01 + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize=14,
                    ha='center', va='bottom')


def registro():
    opcoes2 = ["LogLoss", "F1", "Acurária"]
    paginaselecionada2 = st.sidebar.selectbox("Visualização", opcoes2)
    if(paginaselecionada2==opcoes2[0]):
        registro_logloss()
    elif (paginaselecionada2==opcoes2[1]):
        registro_f1()
    elif(paginaselecionada2 == opcoes2[2]):
        registro_accuracy()


def operalization():
    opcoes2 = ["ROC", "Acurária", 'Metricas', 'Erros e Acertos']
    paginaselecionada2 = st.sidebar.selectbox("Visualização", opcoes2)
    if (paginaselecionada2 == opcoes2[0]):
        operacao_roc()
    elif(paginaselecionada2 == opcoes2[1]):
        operation_accuracy()
    elif(paginaselecionada2 == opcoes2[2]):
        opertion_acuracia_recall_prec()
    elif(paginaselecionada2 == opcoes2[3]):
        operation_error_acertos()

def operacion_metrics():
    df_ml = get_mlflow_experiments()
    df_ml_fh = df_ml[df_ml['status'] == 'FINISHED'].copy()
    df_ml_fh_rv = df_ml_fh[df_ml_fh['tags.mlflow.runName'] == 'RevalidaçãoModelo'].copy()
    df_ml_fh_rv = df_ml_fh_rv[df_ml_fh_rv['metrics.Accuracy'].notnull()].reset_index()
    df_ml_fh_rv = df_ml_fh_rv[df_ml_fh_rv['metrics.Version'].notnull()].reset_index()
    return df_ml_fh_rv

if(paginaselecionada==pages[0]):
    inicial()

elif((paginaselecionada==pages[1])):
    registro()

elif((paginaselecionada==pages[2])):
    operalization()

#Diagrama ML
elif(paginaselecionada==pages[3]):
    st.image("../Docs/Data_Report/Images/Diagrama ML.png")

def dadosa():
    pass


def diagrama_ml():
    st.title('Diagrama ML')

