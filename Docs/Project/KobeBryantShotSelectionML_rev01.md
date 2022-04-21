# Kobe Bryant Shot Selection ML

Este é um projeto de Engenharia de Machine Learning e tem o objetivo de utilizar a base de dados kobe-bryant-shot-selection disponível no
site Kaggle, URL: https://www.kaggle.com/c/kobe-bryant-shot-selection/data. Essa base de dados trás informações como circustâncias e localização, entre outras,
dos arremessos realizados pelo astro da NBA Kobe Bryant durante sua carreira. A intenção é determinar através dos algoritmos de machine learning
de foi convertida a cesta, variável alvo shot_made_flag.

# 1. Repositório e Template

Este projeto possue o seguinte repositório de dados URL: https://github.com/eriktavares/KobeBryantShotSelectionML. As estruturas de diretórios de arquivos foram baseadas no padrão Framework TDSP da Microsoft, e foi baixo o template pela URL https://github.com/Azure/Azure-TDSP-ProjectTemplate. Somente a pasta Simple_Data foi renomeada para Data, por conta da descrição que foi solicitado no enunciado da atividade (moodle). O arquivo de dados foi renomeado para 


```python
import os
import warnings
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree, preprocessing, metrics, model_selection
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
```

Estrutura dos diretórios dentro do Repositório

![tdsp-dir-structure.png](attachment:tdsp-dir-structure.png)

# 2. Diagrama MLOps

Para inicio do projeto de Machine Learning é preciso pensar nos processos necessários para a execução. Dessa forma, é preciso haver inicialmente um Entendimento de Negócio, está etapa é uma visão das atividades macro que envolve o projeto. Esse entendimento da origem a outras 3 atividades principais.

O Planejamento, onde será feito o diagrama pepiline do projeto, entre outros documentos, cronogramas, ferramentos de gestão e acompanhamento. Estrutura de repositórios de código, dados, etc. Ferramentas utilizadas para o desenvolvimento, frameworks, plataformas e etc.

Exploração de dados. os projetos de Machine Learning são baseados em informações, então, dessa forma dependendo do tamanho do projeto, pode ou não ser continua e deve fazer interface com planejamento e entendimento do negócio. E por ultimo abastecer uma base de dados com informações. No caso desse trabalho especifico, essa base é coletada do site kaggle.

Desenvolvimento do Experimento. Com os dados em mãos, é iniciado o processo experimental onde é realizada a modelagem. Esta etapa começa com a validações dos dados, pode ser feita por exemplo utilizando o Pycaret Setup. A preparação dos dados, onde neste trabalho são removidos os dados nulos, normalização, entre outros processos de tratamento. Treino e avaliação, caso os resultados não sejam coonforme esperado, processo de melhoria e otimização, e nova avaliação. Posteriormente o registro, versionamento e depployment do modelo. Claro que podem ocorrer versionamentos durante qualquer etapa desenvolvimento. Após o deployment o modelo entra em operação, e pode ser como uma API, código, serviço, entre outros. A operação é monitorada e pode gerar novos gatilhos de desenvolvimento e melhorias, retreinamentos, etc. 

![Diagrama%20ML.png](attachment:Diagrama%20ML.png)

# 3. Pepilines

Assim como em outros processos de desenvolvimento de software os pepilines também são muito importantes no desenvolvimento de aplicações de machine learning. O uso de pepiline, permite a criação de um fluxo de tarefas a serem seguindas que garantem a automatização de todo o processo de desenvolvimento. Como um algoritmo do processo de trabalho, passando por todas as etapas e que podem ser continuas. Os pipelines de ML são definições portáteis e reproduzíveis de fluxos de trabalho.
O diagrama acima demostra as etapas principais para um modelo de machine learning. Os beneficios da utilização
dos pepilines são diversas, entre elas, automação do processo, desenvolvimento agil, continuo e com qualidade, reprodutibilidade e auditabilidade. Então cada etapa do processo pode ser implementado como um pepiline


# 4. Ferramentas

# Exploração dos Dados

Neste topico extra, pode ser incluído ferramentas de exploração de informação, ou seja dados, para alimentar a base de dados do modelo. 

# Pycaret

Nesse processo de Auto ML, uma ferramento muito importante e que tras inumeros beneficios é o Pycaret. PyCaret é uma biblioteca de aprendizado de máquina de código aberto e de baixo código em Python que 
automatiza fluxos de trabalho de aprendizado de máquina (https://pycaret.gitbook.io/docs/). O Pycaret possui funções para os processos
de preparação dos dados, Treinamentos de modelos, ajuste de hiperparâmetros, analise e interpretação, seleção de modelos e gestão de experiemto.
Dessa forma, utilizar essas funções ja desenvolvidas e testadas gera automação do processo de modelagem que será feito a seguir.
Durante o rastreamento dos experimentos serão utilizadas as funções como a Setup()

Comparado com outras bibliotecas de aprendizado de máquina de código aberto, o PyCaret é uma biblioteca alternativa de baixo código que pode ser usada para substituir centenas de linhas de código por apenas algumas linhas. Isso torna os experimentos exponencialmente rápidos e eficientes. O PyCaret é essencialmente um wrapper Python em torno de várias bibliotecas e estruturas de aprendizado de máquina, como scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray e mais alguns. (https://pycaret.gitbook.io/docs/)


O Pycare possui diversas funções, abaixo esta sendo exemplificado algumas funçãoes que em relação as etapas definidas no processo de experiemntação e desenvolvimento de modelos de machine learning.

Rastreio de Experimentos
Setup()
Essa função inicializa o experimento no PyCaret e prepara o pipeline de transformação com base em todos os parâmetros passados na função. A função de configuração deve ser chamada antes de executar qualquer outra função. Requer apenas dois parâmetros: dados e destino. Todos os outros parâmetros são opcionais (https://pycaret.gitbook.io/docs/get-started/functions/initialize#setting-up-environment).


Treinamento:



compare_model()

Essa função treina e avalia o desempenho de todos os estimadores disponíveis na biblioteca de modelos usando validação cruzada.
Dessa forma o processo de comparação dentre modelos para escolha do melhor podelo fica automatizada e pratica.

create_model()
Essa função treina e avalia o desempenho de um determinado estimador usando validação cruzada. Facilita o treinamento e a busca utilizando a validação cruzada com avaliação de desempenho.

Monitoramento

plot_model()
Esta função analisa o desempenho de um modelo treinado no conjunto hold-out. 

Atualização
calibrate_model()
optimize_threshold

tune_model()
Esta função ajusta os hiperparâmetros do modelo

Provisionamento(Deployment)

Funções como save_model()

Essa função salva o pipeline de transformação e um objeto de modelo treinado no diretório de trabalho atual como um arquivo pickle para uso posterior.

deploy_model()

Essa função implanta todo o pipeline de ML na nuvem.

https://pycaret.gitbook.io/docs/get-started/functions/train

# MLFLOW

Para realizar o gerenciamento do ciclo de vida deste projeto de machine learning, será utilizado o MLFLOW. Conforme a descrição do site "O MLflow é uma plataforma de código aberto para gerenciar o ciclo de vida do ML, incluindo experimentação, reprodutibilidade, implantação e um registro de modelo central. Atualmente, o MLflow oferece quatro componentes: " https://mlflow.org/
- MLflow Tracking

Gravar e consultar experimentos: código, dados, configuração e resultados.

Na etapa de preparção dos dados, são utilizadas algumas funções como log_param e log_metric para gerar o log de parametros, seleção e features e de metricas, que são os tamanhos das bases, dados nulos e etc.

Também as metricas e parametros nos processos de treino e teste dos modelos.

No monitoramento da saúde do modelo, durante a operação, será comparado resultados dos experiementos que geraram o registro do modelo, com os de operação para identificação se a performance do modelo esta se mantendo.


- MLflow Projects

Empacote o código de ciência de dados em um formato para reproduzir execuções em qualquer plataforma.

Permite o monitoramento do modelo e revalidações em outros ambientes.


- MLflow Models

Implanta modelos de aprendizado de máquina em diversos ambientes de atendimento.

Esses pacotes vão ajudar criar um servidor de aplicação do modelo
para requisições Http, via JSON, por exemplo, entre outros formatos possívels.

- Model Registry


Armazena gerencie modelos em um repositório central

Esses pacotes setão utilizados para colocar o modelo em Staging\Produção


Set up do MLFlow Server, executado no notebook MFLOWSetup

# Streamlit

O Streamlit é uma biblioteca Python de código aberto que facilita a criação e o compartilhamento de aplicativos da Web personalizados e bonitos para aprendizado de máquina e ciência de dados. Em apenas alguns minutos, você pode criar e implantar aplicativos de dados poderosos https://docs.streamlit.io/

Todas as etapas do processo podem ser disponibilizados como indicadores visuais no stremlit, através principalmente da criação de graficos interativos.

Experiment Tracking. Pode ser utilizados vizualizações dessa estapa, por exemplo, informações sobre os dados, e sobre os tratamentos utilizados.

Treino e Teste. Pode ser disponibilizado graficos e para monitoramento dessa etapa, até mesmo aproveitamento os proprios artefatos gerados pelo MLFlow.

Monitoramento da saúde. Esse passo será implementado realizando a comparação de metricas do registro com de operação
com visualização dentro do Streamlit

Atualização do modelo. Pode ser comparado com os processos de registros anteriores e gerar visualizações no Streamlit..

e No Deployment podem ser gerados visualizações de metricas relacionadas as versões entre outras.

# Sklearn

No projeto será utilizada a biblioteca SKlearn que possui diversas funções de código de machine learning prontas para a utilização. Essas funções são utilizadas também dentro do Pycaret, conforme consta na propria descrição do pycaret. Também
existem outras bibliotecas que além do sklearn.

Experiment Tracking por exemplo, funções de tratamento de dados e metricas do sklearn serão utilizadas em conjunto com pycaret e mlflow.
Treinamento. As funções utilizadas pelo pycaret serão as do sklearn para treino e teste, como Regressão Logistica e Arvore de decisão.
Monitoramento da Saúde do Modelo. Funções de metricas do sklearn principamente, no nosso caso, principalmente o Log Loss e o F1, mas diversas outras serão registradas nos MFlow.
Atualização do Modelo, Entra novamente funções de treino teste e metricas utilizadas dentro do pycaret, por exemplo.
Deployment. Os algoritmos da bibllioteca treinados e prontos para uso nas diversas formas de aplicaçõe possíveis.

# 5. Artefatos

Aqui está sendo definido o experimento para log dentro do MLFlow, os dados do experimento serão armazenados no banco mlruns.db,
e será utilizado o SQLite como banco de dados. O experimento foi definido como 'Kobe_Bryant_Shot_Experiment', o banco esta hospedado na pasta Code, conforme os exemplos vistos. Os artefatos estão definidos para o diretório ./mlruns a partir da pasta Code.

Esta arquitetura esta representada no Cenário 02 da documentação do MLFlow https://www.mlflow.org/docs/latest/tracking.html#scenario-2-mlflow-on-localhost-with-sqlite. Onde os artefatos são armazenados na pasta mlruns e as entidades no SQL Lite


![scenario_2.png](attachment:scenario_2.png)


```python
#!mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host 127.0.0.1
```

Caso fosse definido um diretório para o armazenamento do banco e artefatos a patir na pasta dos notebook, será preciso incluir 
sqlite:///./mydirectory/mlruns.db no comando. Existem diversos outros cenários que podem ser utilizados na documentação do MLFlow.

PreparacaoDados:

São gerados alguns artefatos, a base  ../Data/Processed/data_filtered.parquet, ../Data/Operalization/base_train_test.parquet' e '../Data/Operalization/base_operation.parquet'. Ambas descritas na etapa abaixo. O objetivo é guardar os dados após a realização 
da seleção de features, e do tratamento de dados nulos por exemplo. Esses arquivos estão disponíveis para ser lidos nessas pastas por exemplo, pelo streamlit ou por outras ferramentos, e no decorrer dos experimentos podem ser ligos, não ficando salvos apenas em tempo de execução do kernel do jupyter.

Treinamento:

Durante o treinamento são gerados um artefato Transformation Pipeline.pkl contendo as informações da execução. e são logados parâmetros resultantes da analise feita e tags.

Durante a criação do modelo são gerados artefatos MLmodel, condayaml, model.pkl, requiriments.etx, alem de imagens dos plot das metricas (extra). Esses artefatos servem para caso preciso realizar a utilização do modelo, por exemplo como Python.
Também esta incluído um dataset ../Data/Operalization/base_operation_processed.parquet' após a o processamento dos resultados com a coluna dos valores que foram preditos para facilitar os calculos de metricas no Streamlit


```python
# Para usar o sqlite como repositorio

mlflow.set_tracking_uri("sqlite:///mlruns.db")

experiment_name = 'Kobe_Bryant_Shot_Experiment'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id
```

# 6. PreparacaoDados

A preparação dos dados é um passo importante no processo de Auto ML, nesta etapa será carregado o tratado os dados para os processos 

seguintes. Os dados são carregados do arquivo ../Data/kobe_dataset.csv que veio do site kaggle, o tamanho inicial foi registrado
com o nome Tamanho/Linhas - Base Entrada. 
A variável alvo 'shot_made_flag' está com dados faltantes, a quantidade de linhas foi registrado como Quantidade de shot_made_flag Faltante

O tamanho resultante da remoção dos dados faltantes foi registrado como Tamanho/Linhas - Base sem dados faltantes.
Posteriormente foi filtrada para somente os dados com arremeços de 2 pontos 2PT Field Goal e salvo em ../Data/Processed/data_filtered.parquet

Essa base foi separada em treino/teste 80% e 20% para operação. Registrados os tamanhos no MLFlow como Tamanho/Linhas - Base Treino/Teste e Tamanho/Linhas - Base Operação, respectivamente. Salvos em '../Data/Operalization/base_train_test.parquet' e
'../Data/Operalization/base_operation.parquet'

Os dados com arremeços de 3 pontos 3PT Field Goal, registrado o tamanho Tamanho/Linhas - Base Novidade
e armazenado em '../Data/Operalization/base_novelty.parquet'.


Name	Value
Quantidade de shot_made_flag Faltante	5000

Tamanho/Linhas - Base Entrada	30697

Tamanho/Linhas - Base Novidade	5412

Tamanho/Linhas - Base Operação	4057

Tamanho/Linhas - Base Treino/Teste	16228

Tamanho/Linhas - Base sem dados faltantes	25697


Essa separação dos dados treino/teste foi feita utilizando shuffle=True para que seja feito de forma aleatória, e o parâmetro stratify array como default, garantido que seja aleatória e estratificada. Aleatória que os dados serão misturados, e o estratificado garante a proporcionalidade das amostras. Essa técnica evita que os dados sejam divididos de forma a não expressão
a real exencia da informação. Por exemplo, se todos os dados de cesta convertidos estivessem no inicio do dataset ou os erros no final, uma divisão mantendo essa ordenação, iria disponibilizar para o modelo, informação agrupopada com uma tendência predominando, diferente dos dados totais. Dessa forma a modelagem ficaria prejudidada, assim como a validação com os dados de teste. Assim, a aleatóriedade e a manutenção das proporcionalizade na divisão dos dados, garante que o modelo esta recebendo a informação coerênte a totalizadade dos dados.    


```python
# COLOCAR RUN DE LEITURA DE DADOS
# PARAMETROS: top_features,
# METRICS: SHAPE de cada base de dados
# ARTIFACTS: nenhum

import warnings
warnings.filterwarnings('ignore')
top_features = ['lat','lon', 'minutes_remaining' , 'period', 'playoffs', 'shot_distance']
target_col = 'shot_made_flag'
target_col_label = 'shot_made_label'

with mlflow.start_run(experiment_id=experiment_id, run_name = 'PreparacaoDados', nested=True):
    
    #Leitura de dados
    path_kb_data_input= '../Data/kobe_dataset.csv'
    df_kb_all = pd.read_csv(path_kb_data_input)
    mlflow.log_metric("Tamanho/Linhas - Base Entrada", df_kb_all.shape[0])
    
    #Descrição Variável alvo
    mapa ={0 : 'Errou', 1 : 'Cesta'}
    df_kb_all['shot_made_label'] = pd.DataFrame(df_kb_all [target_col].map(mapa))
    df_kb_all[[target_col, target_col_label]]
    
    
    #Remoção de dados Faltantes na Shot_made_Flag
    mlflow.log_metric("Quantidade de {} Faltante".format(target_col), df_kb_all['shot_made_flag'].isnull().sum())
    df_kb = df_kb_all[df_kb_all['shot_made_flag'].notnull()].reset_index()
    df_kb[target_col] = df_kb[target_col].astype(int)
    mlflow.log_metric("Tamanho/Linhas - Base sem dados faltantes", df_kb.shape[0])
    
    
    #Seleção de Features
    df_kb_tf = df_kb [top_features + ['shot_type', target_col]].copy()
    mlflow.log_param("top_features", top_features)
    
    #Filtro 2PT Field Goal
    
    df_kb_2PT = df_kb_tf[df_kb_tf['shot_type'] == '2PT Field Goal'].copy().drop('shot_type', axis=1)
    df_kb_2PT.to_parquet('../Data/Processed/data_filtered.parquet')
    
    
    # Separação da base com 80%/20% test_size=0.2
    #stratifyarray-like, default=None If not None, data is split in a stratified fashion, using this as the class labels.
    #shuffle = True
    df_kb_tt, df_kb_operation, ytrain, ytest = model_selection.train_test_split(df_kb_2PT, 
                                                                            df_kb_2PT[target_col],
                                                                            test_size=0.2,
                                                                            shuffle=True)
    
    mlflow.log_param("Percentual Operação", '0.2')
    df_kb_tt[target_col]      = ytrain
    df_kb_operation[target_col] = ytest
    
    
    mlflow.log_metric("Tamanho/Linhas - Base Treino/Teste", df_kb_tt.shape[0])
    mlflow.log_metric("Tamanho/Linhas - Base Operação", df_kb_operation.shape[0])
    
    
    #Base  3PT Field Goal
    df_kb_novelty = df_kb[df_kb['shot_type'] == '3PT Field Goal'].copy().drop('shot_type', axis=1)
    mlflow.log_metric("Tamanho/Linhas - Base Novidade", df_kb_novelty.shape[0]) 
    
    #Envio datasets para "/Data/operalization/base_{train|test}.parquet
    df_kb_tt.to_parquet('../Data/Operalization/base_train_test.parquet')
    df_kb_operation.to_parquet('../Data/Operalization/base_operation.parquet')
    df_kb_novelty.to_parquet('../Data/Operalization/base_novelty.parquet')
    
#label_map = df_wine[['target', 'target_label']].drop_duplicates()
#drop_cols = ['target_label']
#df_wine.drop(drop_cols, axis=1, inplace=True)
#print(df_kb.shape)

#df_kb.head()
#df_kb.keys()
    
    
mlflow.end_run()
```

# 7.Treinamento

Essa função inicializa o experimento no PyCaret e cria o pipeline de transformação com base em todos os parâmetros passados ​​na função. A função de configuração deve ser chamada antes de executar qualquer outra função. São necessários dois parâmetros obrigatórios: data e destino. Todos os outros parâmetros são opcionais. https://pycaret.gitbook.io/docs/get-started/functions/initialize#setting-up-environment.

Neste caso os parâmetros obrigatórios são df_kb_tt (base de dados) e o nome da coluna da váriável alvo.

Os parâmetros para gerar os logs do experimento no MLFLOW.

    - log_experiment = True, 
    - experiment_name = experiment_name, 
    - log_plots = True


As metricas default do Pycaret são: 'Accuracy' 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'. Porém será adicionado também
a Metrica Perda de Log

Perda de log, também conhecida como perda logística ou perda de entropia cruzada.

Esta é a função de perda usada na regressão logística (multinomial) e em suas extensões, como redes neurais, definida como a probabilidade logarítmica negativa de um modelo logístico que retorna probabilidades y_pred para seus dados de treinamento y_true. A perda de log é definida apenas para dois ou mais rótulos
Adicionando Metric Loss Log. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

# 7a - Regressão Logistica


```python
#import pycaret.classification as pc
# COLOCAR RUN DE TREINAMENTO DE MODELOS
# PARAMETROS: fold_strategy, fold, model_name, registered_model_name, cross_validation
# METRICS: auto sklearn
# ARTIFACTS: plots
# add Log Loss metric in pycaret
import pycaret.classification as pc
from sklearn.metrics import log_loss

registered_model_name = 'modelo_regressão_kb'
model_name = 'lr'
probability_threshold = 0.5
cross_validation = True
fold_strategy = 'stratifiedkfold',
fold = 10
with mlflow.start_run(experiment_id=experiment_id, run_name = 'Treinamento', nested=True):
    # train/test
    s = pc.setup(data = df_kb_tt, 
                 target = target_col,
                 train_size=0.7,
                 silent = True,
                 fold_strategy = 'stratifiedkfold',
                 fold = fold,
                 log_experiment = True, 
                 experiment_name = experiment_name, 
                 log_plots = True
                )
    pc.add_metric('logloss', 'LogLoss', log_loss, greater_is_better=False)
    bestmodel = pc.create_model(model_name,
                                cross_validation = cross_validation,
                                probability_threshold=probability_threshold)

    # Log do run, e nao do modelo respectivo
    classification_plots = [ 'auc','pr','confusion_matrix',
                          #'error', 'class_report', 
                        'threshold', 'f1', 'logloss',
                         'learning','vc','feature',
                       ]
    for plot_type in classification_plots:
        print('=> Aplicando plot ', plot_type)
        try:
            artifact = pc.plot_model(bestmodel, plot=plot_type, save=True, use_train_data=False)
            mlflow.log_artifact(artifact)
        except:
            print('=> Nao possivel plotar: ', plot_type )
            continue

    #pc.save_model(bestmodel, f'./{registered_model_name}') 
    # Carrega novamente o pipeline + bestmodel
    #model_pipe = pc.load_model(f'./{registered_model_name}')


mlflow.end_run()
```

    INFO:logs:Saving 'Feature Importance.png'
    INFO:logs:Visual Rendered Successfully
    INFO:logs:plot_model() succesfully completed......................................
    


```python
pc.get_metrics() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Display Name</th>
      <th>Score Function</th>
      <th>Scorer</th>
      <th>Target</th>
      <th>Args</th>
      <th>Greater is Better</th>
      <th>Multiclass</th>
      <th>Custom</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc</th>
      <td>Accuracy</td>
      <td>Accuracy</td>
      <td>&lt;function accuracy_score at 0x0000022712FD68C8&gt;</td>
      <td>accuracy</td>
      <td>pred</td>
      <td>{}</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>auc</th>
      <td>AUC</td>
      <td>AUC</td>
      <td>&lt;function roc_auc_score at 0x0000022712FD1268&gt;</td>
      <td>make_scorer(roc_auc_score, needs_proba=True, e...</td>
      <td>pred_proba</td>
      <td>{'average': 'weighted', 'multi_class': 'ovr'}</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>Recall</td>
      <td>Recall</td>
      <td>&lt;pycaret.internal.metrics.BinaryMulticlassScor...</td>
      <td>make_scorer(recall_score, average=macro)</td>
      <td>pred</td>
      <td>{'average': 'macro'}</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>Precision</td>
      <td>Prec.</td>
      <td>&lt;pycaret.internal.metrics.BinaryMulticlassScor...</td>
      <td>make_scorer(precision_score, average=weighted)</td>
      <td>pred</td>
      <td>{'average': 'weighted'}</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>F1</td>
      <td>F1</td>
      <td>&lt;pycaret.internal.metrics.BinaryMulticlassScor...</td>
      <td>make_scorer(f1_score, average=weighted)</td>
      <td>pred</td>
      <td>{'average': 'weighted'}</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>kappa</th>
      <td>Kappa</td>
      <td>Kappa</td>
      <td>&lt;function cohen_kappa_score at 0x0000022712FD6...</td>
      <td>make_scorer(cohen_kappa_score)</td>
      <td>pred</td>
      <td>{}</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>mcc</th>
      <td>MCC</td>
      <td>MCC</td>
      <td>&lt;function matthews_corrcoef at 0x0000022712FE3...</td>
      <td>make_scorer(matthews_corrcoef)</td>
      <td>pred</td>
      <td>{}</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>logloss</th>
      <td>LogLoss</td>
      <td>LogLoss</td>
      <td>&lt;function log_loss at 0x0000022712FE8378&gt;</td>
      <td>make_scorer(log_loss, greater_is_better=False)</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred_holdout = pc.predict_model(bestmodel)
```

    INFO:logs:Initializing predict_model()
    INFO:logs:predict_model(drift_kwargs=None, display=None, ml_usecase=MLUsecase.CLASSIFICATION, verbose=True, round=4, raw_score=False, drift_report=False, encoded_labels=False, probability_threshold=None, estimator=CustomProbabilityThresholdClassifier(C=1.0, class_weight=None,
                                         classifier=LogisticRegression(C=1.0,
                                                                       class_weight=None,
                                                                       dual=False,
                                                                       fit_intercept=True,
                                                                       intercept_scaling=1,
                                                                       l1_ratio=None,
                                                                       max_iter=1000,
                                                                       multi_class='auto',
                                                                       n_jobs=None,
                                                                       penalty='l2',
                                                                       random_state=4678,
                                                                       solver='lbfgs',
                                                                       tol=0.0001,
                                                                       verbose=0,
                                                                       warm_start=False),
                                         dual=False, fit_intercept=True,
                                         intercept_scaling=1, l1_ratio=None,
                                         max_iter=1000, multi_class='auto',
                                         n_jobs=None, penalty='l2',
                                         probability_threshold=0.5,
                                         random_state=4678, solver='lbfgs',
                                         tol=0.0001, verbose=0, warm_start=False))
    INFO:logs:Checking exceptions
    INFO:logs:Preloading libraries
    INFO:logs:Preparing display monitor
    


<style  type="text/css" >
</style><table id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>        <th class="col_heading level0 col7" >MCC</th>        <th class="col_heading level0 col8" >LogLoss</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0row0_col0" class="data row0 col0" >Logistic Regression</td>
                        <td id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0row0_col1" class="data row0 col1" >0.5701</td>
                        <td id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0row0_col2" class="data row0 col2" >0.5859</td>
                        <td id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0row0_col3" class="data row0 col3" >0.4945</td>
                        <td id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0row0_col4" class="data row0 col4" >0.5387</td>
                        <td id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0row0_col5" class="data row0 col5" >0.5156</td>
                        <td id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0row0_col6" class="data row0 col6" >0.1305</td>
                        <td id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0row0_col7" class="data row0 col7" >0.1309</td>
                        <td id="T_9a9f6e0c_c179_11ec_95f7_00d76d400aa0row0_col8" class="data row0 col8" >14.8471</td>
            </tr>
    </tbody></table>


# 7C - Arvore de Decisão


```python
import pycaret.classification as pc
# COLOCAR RUN DE TREINAMENTO DE MODELOS
# PARAMETROS: fold_strategy, fold, model_name, registered_model_name, cross_validation
# METRICS: auto sklearn
# ARTIFACTS: plots
# add Log Loss metric in pycaret




registered_model_name = 'modelo_arvore_kb'
model_name = 'dt'
probability_threshold = 0.5
cross_validation = True
fold_strategy = 'stratifiedkfold',
fold = 10
with mlflow.start_run(experiment_id=experiment_id, run_name = 'Treinamento', nested=True):
    # train/test
   
    #pc.add_metric('logloss', 'LogLoss', log_loss, greater_is_better=False)
    bestmodel = pc.create_model(model_name,
                                cross_validation = cross_validation, 
                                probability_threshold=probability_threshold)

    # Log do run, e nao do modelo respectivo
    #classification_plots = [ 'f1','logloss']
   # for plot_type in classification_plots:
       # print('=> Aplicando plot ', plot_type)
       # try:
        #    artifact = pc.plot_model(bestmodel, plot=plot_type, save=True, use_train_data=False)
        #    mlflow.log_artifact(artifact)
        #except:
         #   print('=> Nao possivel plotar: ', plot_type )
         #   continue

    #pc.save_model(bestmodel, f'./{registered_model_name}') 
    # Carrega novamente o pipeline + bestmodel
    #model_pipe = pc.load_model(f'./{registered_model_name}')


mlflow.end_run()
```


<style  type="text/css" >
#T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col0,#T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col1,#T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col2,#T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col3,#T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col4,#T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col5,#T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col6,#T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col7{
            background:  yellow;
        }</style><table id="T_9d72c474_c179_11ec_80f0_00d76d400aa0" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>        <th class="col_heading level0 col7" >LogLoss</th>    </tr>    <tr>        <th class="index_name level0" >Fold</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row0_col0" class="data row0 col0" >0.5326</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row0_col1" class="data row0 col1" >0.5102</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row0_col2" class="data row0 col2" >0.6062</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row0_col3" class="data row0 col3" >0.5154</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row0_col4" class="data row0 col4" >0.5571</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row0_col5" class="data row0 col5" >0.0691</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row0_col6" class="data row0 col6" >0.0701</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row0_col7" class="data row0 col7" >16.1447</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row1_col0" class="data row1 col0" >0.5546</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row1_col1" class="data row1 col1" >0.5335</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row1_col2" class="data row1 col2" >0.6171</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row1_col3" class="data row1 col3" >0.5354</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row1_col4" class="data row1 col4" >0.5734</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row1_col5" class="data row1 col5" >0.1123</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row1_col6" class="data row1 col6" >0.1135</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row1_col7" class="data row1 col7" >15.3846</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row2_col0" class="data row2 col0" >0.5423</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row2_col1" class="data row2 col1" >0.5156</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row2_col2" class="data row2 col2" >0.5935</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row2_col3" class="data row2 col3" >0.5249</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row2_col4" class="data row2 col4" >0.5571</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row2_col5" class="data row2 col5" >0.0872</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row2_col6" class="data row2 col6" >0.0879</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row2_col7" class="data row2 col7" >15.8102</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row3_col0" class="data row3 col0" >0.5079</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row3_col1" class="data row3 col1" >0.4807</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row3_col2" class="data row3 col2" >0.5917</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row3_col3" class="data row3 col3" >0.4939</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row3_col4" class="data row3 col4" >0.5384</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row3_col5" class="data row3 col5" >0.0206</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row3_col6" class="data row3 col6" >0.0210</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row3_col7" class="data row3 col7" >16.9960</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row4_col0" class="data row4 col0" >0.5423</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row4_col1" class="data row4 col1" >0.5248</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row4_col2" class="data row4 col2" >0.5826</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row4_col3" class="data row4 col3" >0.5254</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row4_col4" class="data row4 col4" >0.5525</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row4_col5" class="data row4 col5" >0.0866</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row4_col6" class="data row4 col6" >0.0871</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row4_col7" class="data row4 col7" >15.8102</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row5_col0" class="data row5 col0" >0.5211</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row5_col1" class="data row5 col1" >0.4963</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row5_col2" class="data row5 col2" >0.5644</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row5_col3" class="data row5 col3" >0.5057</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row5_col4" class="data row5 col4" >0.5334</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row5_col5" class="data row5 col5" >0.0446</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row5_col6" class="data row5 col6" >0.0449</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row5_col7" class="data row5 col7" >16.5399</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row6_col0" class="data row6 col0" >0.5625</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row6_col1" class="data row6 col1" >0.5380</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row6_col2" class="data row6 col2" >0.6298</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row6_col3" class="data row6 col3" >0.5422</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row6_col4" class="data row6 col4" >0.5827</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row6_col5" class="data row6 col5" >0.1283</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row6_col6" class="data row6 col6" >0.1299</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row6_col7" class="data row6 col7" >15.1109</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row7_col0" class="data row7 col0" >0.5335</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row7_col1" class="data row7 col1" >0.5061</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row7_col2" class="data row7 col2" >0.5662</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row7_col3" class="data row7 col3" >0.5174</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row7_col4" class="data row7 col4" >0.5407</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row7_col5" class="data row7 col5" >0.0686</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row7_col6" class="data row7 col6" >0.0689</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row7_col7" class="data row7 col7" >16.1142</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row8_col0" class="data row8 col0" >0.5484</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row8_col1" class="data row8 col1" >0.5197</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row8_col2" class="data row8 col2" >0.6123</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row8_col3" class="data row8 col3" >0.5306</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row8_col4" class="data row8 col4" >0.5685</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row8_col5" class="data row8 col5" >0.0999</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row8_col6" class="data row8 col6" >0.1010</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row8_col7" class="data row8 col7" >15.5974</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row9_col0" class="data row9 col0" >0.5427</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row9_col1" class="data row9 col1" >0.5229</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row9_col2" class="data row9 col2" >0.5808</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row9_col3" class="data row9 col3" >0.5263</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row9_col4" class="data row9 col4" >0.5522</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row9_col5" class="data row9 col5" >0.0874</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row9_col6" class="data row9 col6" >0.0878</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row9_col7" class="data row9 col7" >15.7937</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col0" class="data row10 col0" >0.5388</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col1" class="data row10 col1" >0.5148</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col2" class="data row10 col2" >0.5944</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col3" class="data row10 col3" >0.5217</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col4" class="data row10 col4" >0.5556</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col5" class="data row10 col5" >0.0804</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col6" class="data row10 col6" >0.0812</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row10_col7" class="data row10 col7" >15.9302</td>
            </tr>
            <tr>
                        <th id="T_9d72c474_c179_11ec_80f0_00d76d400aa0level0_row11" class="row_heading level0 row11" >Std</th>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row11_col0" class="data row11 col0" >0.0151</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row11_col1" class="data row11 col1" >0.0164</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row11_col2" class="data row11 col2" >0.0206</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row11_col3" class="data row11 col3" >0.0135</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row11_col4" class="data row11 col4" >0.0150</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row11_col5" class="data row11 col5" >0.0299</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row11_col6" class="data row11 col6" >0.0303</td>
                        <td id="T_9d72c474_c179_11ec_80f0_00d76d400aa0row11_col7" class="data row11 col7" >0.5210</td>
            </tr>
    </tbody></table>


    INFO:logs:create_model_container: 2
    INFO:logs:master_model_container: 2
    INFO:logs:display_container: 4
    INFO:logs:CustomProbabilityThresholdClassifier(ccp_alpha=0.0, class_weight=None,
                                         classifier=DecisionTreeClassifier(ccp_alpha=0.0,
                                                                           class_weight=None,
                                                                           criterion='gini',
                                                                           max_depth=None,
                                                                           max_features=None,
                                                                           max_leaf_nodes=None,
                                                                           min_impurity_decrease=0.0,
                                                                           min_impurity_split=None,
                                                                           min_samples_leaf=1,
                                                                           min_samples_split=2,
                                                                           min_weight_fraction_leaf=0.0,
                                                                           presort='deprecated',
                                                                           random_state=4678,
                                                                           splitter='best'),
                                         criterion='gini', max_depth=None,
                                         max_features=None, max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1, min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         presort='deprecated',
                                         probability_threshold=0.5,
                                         random_state=4678, splitter='best')
    INFO:logs:create_model() succesfully completed......................................
    

# 7C. Escolha Livre

Uma forma de realização de uma escolha para um algoritmo seria utilizar o função compare_models do Pycaret. O sort define o parâmetro de ordenação, nesse caso foi utilizado o Log Loss. Para a escolha do moelhor modelo, pode ser utilizada a função compare_models, e neste caso o melhor resultado foi o Gradient Boosting Classifier e Ada Boost Classifier. Então seria escolhido o Gradient Boosting Classifier. Se for utilizado uma outra metrica para o sort, outro modelo pode ser selecionado como melhor resultado, ou até mesmo em outra simulação.


```python
with mlflow.start_run(experiment_id=experiment_id, run_name = 'Compare', nested=True):
    best_model = pc.compare_models(n_select = 1, sort='logloss')
    mlflow.autolog()
mlflow.end_run()
```


<style  type="text/css" >
    #T_3a44f98c_c189_11ec_b466_00d76d400aa0 th {
          text-align: left;
    }#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col0,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col5,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col8{
            text-align:  left;
            text-align:  left;
        }#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col1,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col4,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col6,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col7,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col8,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col2,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col3,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col5{
            text-align:  left;
            text-align:  left;
            background-color:  yellow;
        }#T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col9,#T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col9{
            text-align:  left;
            text-align:  left;
            background-color:  lightgrey;
        }#T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col9{
            text-align:  left;
            text-align:  left;
            background-color:  yellow;
            background-color:  lightgrey;
        }</style><table id="T_3a44f98c_c189_11ec_b466_00d76d400aa0" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>        <th class="col_heading level0 col7" >MCC</th>        <th class="col_heading level0 col8" >LogLoss</th>        <th class="col_heading level0 col9" >TT (Sec)</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row0" class="row_heading level0 row0" >ada</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col0" class="data row0 col0" >Ada Boost Classifier</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col1" class="data row0 col1" >0.5873</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col2" class="data row0 col2" >0.5957</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col3" class="data row0 col3" >0.3713</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col4" class="data row0 col4" >0.6263</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col5" class="data row0 col5" >0.4658</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col6" class="data row0 col6" >0.1640</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col7" class="data row0 col7" >0.1792</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col8" class="data row0 col8" >14.2547</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row0_col9" class="data row0 col9" >0.1280</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row1" class="row_heading level0 row1" >gbc</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col0" class="data row1 col0" >Gradient Boosting Classifier</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col1" class="data row1 col1" >0.5863</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col2" class="data row1 col2" >0.5947</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col3" class="data row1 col3" >0.3892</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col4" class="data row1 col4" >0.6169</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col5" class="data row1 col5" >0.4771</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col6" class="data row1 col6" >0.1630</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col7" class="data row1 col7" >0.1750</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col8" class="data row1 col8" >14.2881</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row1_col9" class="data row1 col9" >0.3690</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row2" class="row_heading level0 row2" >lda</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col0" class="data row2 col0" >Linear Discriminant Analysis</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col1" class="data row2 col1" >0.5764</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col2" class="data row2 col2" >0.5957</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col3" class="data row2 col3" >0.5044</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col4" class="data row2 col4" >0.5720</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col5" class="data row2 col5" >0.5359</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col6" class="data row2 col6" >0.1491</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col7" class="data row2 col7" >0.1502</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col8" class="data row2 col8" >14.6317</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row2_col9" class="data row2 col9" >0.0310</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row3" class="row_heading level0 row3" >ridge</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col0" class="data row3 col0" >Ridge Classifier</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col1" class="data row3 col1" >0.5762</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col2" class="data row3 col2" >0.0000</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col3" class="data row3 col3" >0.5043</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col4" class="data row3 col4" >0.5718</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col5" class="data row3 col5" >0.5358</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col6" class="data row3 col6" >0.1488</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col7" class="data row3 col7" >0.1498</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col8" class="data row3 col8" >14.6378</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row3_col9" class="data row3 col9" >0.0200</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row4" class="row_heading level0 row4" >lr</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col0" class="data row4 col0" >Logistic Regression</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col1" class="data row4 col1" >0.5754</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col2" class="data row4 col2" >0.5961</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col3" class="data row4 col3" >0.5005</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col4" class="data row4 col4" >0.5714</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col5" class="data row4 col5" >0.5334</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col6" class="data row4 col6" >0.1470</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col7" class="data row4 col7" >0.1482</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col8" class="data row4 col8" >14.6652</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row4_col9" class="data row4 col9" >0.8260</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row5" class="row_heading level0 row5" >lightgbm</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col0" class="data row5 col0" >Light Gradient Boosting Machine</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col1" class="data row5 col1" >0.5648</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col2" class="data row5 col2" >0.5842</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col3" class="data row5 col3" >0.4759</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col4" class="data row5 col4" >0.5608</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col5" class="data row5 col5" >0.5147</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col6" class="data row5 col6" >0.1251</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col7" class="data row5 col7" >0.1265</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col8" class="data row5 col8" >15.0301</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row5_col9" class="data row5 col9" >0.0640</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row6" class="row_heading level0 row6" >rf</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col0" class="data row6 col0" >Random Forest Classifier</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col1" class="data row6 col1" >0.5532</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col2" class="data row6 col2" >0.5604</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col3" class="data row6 col3" >0.5456</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col4" class="data row6 col4" >0.5391</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col5" class="data row6 col5" >0.5422</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col6" class="data row6 col6" >0.1060</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col7" class="data row6 col7" >0.1060</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col8" class="data row6 col8" >15.4315</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row6_col9" class="data row6 col9" >0.4300</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row7" class="row_heading level0 row7" >et</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col0" class="data row7 col0" >Extra Trees Classifier</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col1" class="data row7 col1" >0.5473</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col2" class="data row7 col2" >0.5499</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col3" class="data row7 col3" >0.5632</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col4" class="data row7 col4" >0.5317</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col5" class="data row7 col5" >0.5469</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col6" class="data row7 col6" >0.0954</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col7" class="data row7 col7" >0.0956</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col8" class="data row7 col8" >15.6353</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row7_col9" class="data row7 col9" >0.4970</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row8" class="row_heading level0 row8" >knn</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col0" class="data row8 col0" >K Neighbors Classifier</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col1" class="data row8 col1" >0.5419</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col2" class="data row8 col2" >0.5487</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col3" class="data row8 col3" >0.5086</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col4" class="data row8 col4" >0.5290</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col5" class="data row8 col5" >0.5185</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col6" class="data row8 col6" >0.0819</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col7" class="data row8 col7" >0.0820</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col8" class="data row8 col8" >15.8238</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row8_col9" class="data row8 col9" >0.1270</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row9" class="row_heading level0 row9" >nb</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col0" class="data row9 col0" >Naive Bayes</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col1" class="data row9 col1" >0.5365</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col2" class="data row9 col2" >0.5831</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col3" class="data row9 col3" >0.7159</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col4" class="data row9 col4" >0.5200</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col5" class="data row9 col5" >0.5978</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col6" class="data row9 col6" >0.0828</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col7" class="data row9 col7" >0.0831</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col8" class="data row9 col8" >16.0092</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row9_col9" class="data row9 col9" >0.0160</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row10" class="row_heading level0 row10" >dt</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col0" class="data row10 col0" >Decision Tree Classifier</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col1" class="data row10 col1" >0.5351</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col2" class="data row10 col2" >0.5148</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col3" class="data row10 col3" >0.5812</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col4" class="data row10 col4" >0.5188</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col5" class="data row10 col5" >0.5481</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col6" class="data row10 col6" >0.0726</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col7" class="data row10 col7" >0.0731</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col8" class="data row10 col8" >16.0579</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row10_col9" class="data row10 col9" >0.0360</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row11" class="row_heading level0 row11" >svm</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col0" class="data row11 col0" >SVM - Linear Kernel</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col1" class="data row11 col1" >0.5305</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col2" class="data row11 col2" >0.0000</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col3" class="data row11 col3" >0.5218</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col4" class="data row11 col4" >0.5090</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col5" class="data row11 col5" >0.4251</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col6" class="data row11 col6" >0.0601</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col7" class="data row11 col7" >0.0776</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col8" class="data row11 col8" >16.2159</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row11_col9" class="data row11 col9" >0.0920</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row12" class="row_heading level0 row12" >dummy</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col0" class="data row12 col0" >Dummy Classifier</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col1" class="data row12 col1" >0.5148</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col2" class="data row12 col2" >0.5000</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col3" class="data row12 col3" >0.0000</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col4" class="data row12 col4" >0.0000</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col5" class="data row12 col5" >0.0000</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col6" class="data row12 col6" >0.0000</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col7" class="data row12 col7" >0.0000</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col8" class="data row12 col8" >16.7570</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row12_col9" class="data row12 col9" >0.0180</td>
            </tr>
            <tr>
                        <th id="T_3a44f98c_c189_11ec_b466_00d76d400aa0level0_row13" class="row_heading level0 row13" >qda</th>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col0" class="data row13 col0" >Quadratic Discriminant Analysis</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col1" class="data row13 col1" >0.4981</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col2" class="data row13 col2" >0.4498</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col3" class="data row13 col3" >0.5713</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col4" class="data row13 col4" >0.4839</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col5" class="data row13 col5" >0.5065</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col6" class="data row13 col6" >0.0004</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col7" class="data row13 col7" >0.0024</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col8" class="data row13 col8" >17.3351</td>
                        <td id="T_3a44f98c_c189_11ec_b466_00d76d400aa0row13_col9" class="data row13 col9" >0.0230</td>
            </tr>
    </tbody></table>


    INFO:logs:create_model_container: 43
    INFO:logs:master_model_container: 43
    INFO:logs:display_container: 21
    INFO:logs:AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                       n_estimators=50, random_state=4678)
    INFO:logs:compare_models() succesfully completed......................................
    2022/04/21 11:39:27 INFO mlflow.tracking.fluent: Autologging successfully enabled for sklearn.
    2022/04/21 11:39:27 INFO mlflow.tracking.fluent: Autologging successfully enabled for lightgbm.
    


```python
best_model
```




    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                       n_estimators=50, random_state=4678)




```python
with mlflow.start_run(experiment_id=experiment_id, run_name = 'Tune', nested=True):
    tuned_model = pc.tune_model(best_model,
                            optimize = 'logloss',
                            search_library = 'scikit-learn',
                            search_algorithm = 'random',
                            n_iter = 4)
mlflow.end_run()
```


<style  type="text/css" >
#T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col0,#T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col1,#T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col2,#T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col3,#T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col4,#T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col5,#T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col6,#T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col7{
            background:  yellow;
        }</style><table id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>        <th class="col_heading level0 col7" >LogLoss</th>    </tr>    <tr>        <th class="index_name level0" >Fold</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row0_col0" class="data row0 col0" >0.5854</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row0_col1" class="data row0 col1" >0.5779</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row0_col2" class="data row0 col2" >0.3448</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row0_col3" class="data row0 col3" >0.6333</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row0_col4" class="data row0 col4" >0.4465</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row0_col5" class="data row0 col5" >0.1589</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row0_col6" class="data row0 col6" >0.1778</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row0_col7" class="data row0 col7" >14.3203</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row1_col0" class="data row1 col0" >0.5924</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row1_col1" class="data row1 col1" >0.5966</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row1_col2" class="data row1 col2" >0.3539</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row1_col3" class="data row1 col3" >0.6457</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row1_col4" class="data row1 col4" >0.4572</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row1_col5" class="data row1 col5" >0.1733</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row1_col6" class="data row1 col6" >0.1934</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row1_col7" class="data row1 col7" >14.0771</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row2_col0" class="data row2 col0" >0.5915</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row2_col1" class="data row2 col1" >0.5938</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row2_col2" class="data row2 col2" >0.3775</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row2_col3" class="data row2 col3" >0.6322</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row2_col4" class="data row2 col4" >0.4727</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row2_col5" class="data row2 col5" >0.1727</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row2_col6" class="data row2 col6" >0.1880</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row2_col7" class="data row2 col7" >14.1075</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row3_col0" class="data row3 col0" >0.5871</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row3_col1" class="data row3 col1" >0.5938</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row3_col2" class="data row3 col2" >0.3593</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row3_col3" class="data row3 col3" >0.6306</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row3_col4" class="data row3 col4" >0.4578</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row3_col5" class="data row3 col5" >0.1631</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row3_col6" class="data row3 col6" >0.1800</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row3_col7" class="data row3 col7" >14.2595</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row4_col0" class="data row4 col0" >0.5915</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row4_col1" class="data row4 col1" >0.6101</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row4_col2" class="data row4 col2" >0.3412</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row4_col3" class="data row4 col3" >0.6505</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row4_col4" class="data row4 col4" >0.4476</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row4_col5" class="data row4 col5" >0.1709</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row4_col6" class="data row4 col6" >0.1934</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row4_col7" class="data row4 col7" >14.1075</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row5_col0" class="data row5 col0" >0.5783</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row5_col1" class="data row5 col1" >0.5883</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row5_col2" class="data row5 col2" >0.3503</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row5_col3" class="data row5 col3" >0.6146</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row5_col4" class="data row5 col4" >0.4462</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row5_col5" class="data row5 col5" >0.1453</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row5_col6" class="data row5 col6" >0.1603</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row5_col7" class="data row5 col7" >14.5635</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row6_col0" class="data row6 col0" >0.5995</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row6_col1" class="data row6 col1" >0.6077</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row6_col2" class="data row6 col2" >0.3612</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row6_col3" class="data row6 col3" >0.6589</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row6_col4" class="data row6 col4" >0.4666</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row6_col5" class="data row6 col5" >0.1876</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row6_col6" class="data row6 col6" >0.2094</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row6_col7" class="data row6 col7" >13.8338</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row7_col0" class="data row7 col0" >0.5898</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row7_col1" class="data row7 col1" >0.5896</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row7_col2" class="data row7 col2" >0.3503</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row7_col3" class="data row7 col3" >0.6412</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row7_col4" class="data row7 col4" >0.4531</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row7_col5" class="data row7 col5" >0.1679</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row7_col6" class="data row7 col6" >0.1876</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row7_col7" class="data row7 col7" >14.1683</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row8_col0" class="data row8 col0" >0.6083</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row8_col1" class="data row8 col1" >0.6148</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row8_col2" class="data row8 col2" >0.3822</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row8_col3" class="data row8 col3" >0.6698</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row8_col4" class="data row8 col4" >0.4867</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row8_col5" class="data row8 col5" >0.2066</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row8_col6" class="data row8 col6" >0.2279</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row8_col7" class="data row8 col7" >13.5298</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row9_col0" class="data row9 col0" >0.6035</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row9_col1" class="data row9 col1" >0.6190</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row9_col2" class="data row9 col2" >0.3757</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row9_col3" class="data row9 col3" >0.6613</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row9_col4" class="data row9 col4" >0.4792</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row9_col5" class="data row9 col5" >0.1966</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row9_col6" class="data row9 col6" >0.2172</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row9_col7" class="data row9 col7" >13.6939</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col0" class="data row10 col0" >0.5927</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col1" class="data row10 col1" >0.5992</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col2" class="data row10 col2" >0.3596</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col3" class="data row10 col3" >0.6438</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col4" class="data row10 col4" >0.4614</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col5" class="data row10 col5" >0.1743</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col6" class="data row10 col6" >0.1935</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row10_col7" class="data row10 col7" >14.0661</td>
            </tr>
            <tr>
                        <th id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0level0_row11" class="row_heading level0 row11" >Std</th>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row11_col0" class="data row11 col0" >0.0084</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row11_col1" class="data row11 col1" >0.0125</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row11_col2" class="data row11 col2" >0.0136</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row11_col3" class="data row11 col3" >0.0159</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row11_col4" class="data row11 col4" >0.0136</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row11_col5" class="data row11 col5" >0.0173</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row11_col6" class="data row11 col6" >0.0190</td>
                        <td id="T_8a221e1a_c18b_11ec_a636_00d76d400aa0row11_col7" class="data row11 col7" >0.2904</td>
            </tr>
    </tbody></table>


    INFO:logs:create_model_container: 44
    INFO:logs:master_model_container: 44
    INFO:logs:display_container: 22
    INFO:logs:AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.4,
                       n_estimators=280, random_state=4678)
    INFO:logs:tune_model() succesfully completed......................................
    


```python
with mlflow.start_run(experiment_id=experiment_id, run_name = 'Evaluate Tuned', nested=True):
    calibrated_model = pc.calibrate_model(tuned_model, method='sigmoid', calibrate_fold=5, fold=5)
    pc.plot_model(calibrated_model, plot='calibration')
mlflow.end_run()
```


    
![png](output_51_0.png)
    


    INFO:logs:Visual Rendered Successfully
    INFO:logs:plot_model() succesfully completed......................................
    


```python
pc.optimize_threshold(calibrated_model, optimize = 'logloss');
```

    INFO:logs:Initializing optimize_threshold()
    INFO:logs:optimize_threshold(plot_kwargs=None, return_data=False, grid_interval=0.1, optimize=logloss, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Importing libraries
    INFO:logs:Checking exceptions
    INFO:logs:defining variables
    INFO:logs:starting optimization loop
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.0, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 46
    INFO:logs:master_model_container: 46
    INFO:logs:display_container: 24
    INFO:logs:CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid')
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.1, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 47
    INFO:logs:master_model_container: 47
    INFO:logs:display_container: 25
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=0.1)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.2, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 48
    INFO:logs:master_model_container: 48
    INFO:logs:display_container: 26
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=0.2)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.30000000000000004, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 49
    INFO:logs:master_model_container: 49
    INFO:logs:display_container: 27
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=0.30000000000000004)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.4, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 50
    INFO:logs:master_model_container: 50
    INFO:logs:display_container: 28
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=0.4)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.5, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 51
    INFO:logs:master_model_container: 51
    INFO:logs:display_container: 29
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=0.5)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.6000000000000001, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 52
    INFO:logs:master_model_container: 52
    INFO:logs:display_container: 30
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=0.6000000000000001)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.7000000000000001, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 53
    INFO:logs:master_model_container: 53
    INFO:logs:display_container: 31
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=0.7000000000000001)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.8, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 54
    INFO:logs:master_model_container: 54
    INFO:logs:display_container: 32
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=0.8)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=0.9, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 55
    INFO:logs:master_model_container: 55
    INFO:logs:display_container: 33
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=0.9)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(kwargs={}, return_train_score=False, display=None, probability_threshold=1.0, add_to_model_list=True, experiment_custom_tags=None, metrics=None, system=False, verbose=False, refit=True, groups=None, fit_kwargs=None, predict=True, cross_validation=True, round=4, fold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ada Boost Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:Uploading results into container
    INFO:logs:Uploading model into container now
    INFO:logs:create_model_container: 56
    INFO:logs:master_model_container: 56
    INFO:logs:display_container: 34
    INFO:logs:CustomProbabilityThresholdClassifier(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                           base_estimator=None,
                                                                           learning_rate=0.4,
                                                                           n_estimators=280,
                                                                           random_state=4678),
                                         classifier=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                                                             base_estimator=None,
                                                                                                             learning_rate=0.4,
                                                                                                             n_estimators=280,
                                                                                                             random_state=4678),
                                                                           cv=5,
                                                                           method='sigmoid'),
                                         cv=5, method='sigmoid',
                                         probability_threshold=1.0)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:optimization loop finished successfully
    INFO:logs:plotting optimization threshold using plotly
    INFO:logs:Figure ready for render
    


<div>                            <div id="3edf2bce-2a5a-4079-bd15-eadb047d8de1" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("3edf2bce-2a5a-4079-bd15-eadb047d8de1")) {                    Plotly.newPlot(                        "3edf2bce-2a5a-4079-bd15-eadb047d8de1",                        [{"hovertemplate":"variable=Accuracy<br>probability_threshold=%{x}<br>value=%{y}<extra></extra>","legendgroup":"Accuracy","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"Accuracy","orientation":"v","showlegend":true,"x":[0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0],"xaxis":"x","y":[0.5922,0.4852,0.4852,0.4852,0.4993,0.5922,0.5918,0.5148,0.5148,0.5148,0.5148],"yaxis":"y","type":"scatter"},{"hovertemplate":"variable=AUC<br>probability_threshold=%{x}<br>value=%{y}<extra></extra>","legendgroup":"AUC","line":{"color":"#EF553B","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"AUC","orientation":"v","showlegend":true,"x":[0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0],"xaxis":"x","y":[0.6012,0.6012,0.6012,0.6012,0.6012,0.6012,0.6012,0.6012,0.6012,0.6012,0.6012],"yaxis":"y","type":"scatter"},{"hovertemplate":"variable=Recall<br>probability_threshold=%{x}<br>value=%{y}<extra></extra>","legendgroup":"Recall","line":{"color":"#00cc96","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"Recall","orientation":"v","showlegend":true,"x":[0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0],"xaxis":"x","y":[0.3593,1.0,1.0,1.0,0.9523,0.3593,0.3375,0.0,0.0,0.0,0.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"variable=Prec.<br>probability_threshold=%{x}<br>value=%{y}<extra></extra>","legendgroup":"Prec.","line":{"color":"#ab63fa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"Prec.","orientation":"v","showlegend":true,"x":[0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0],"xaxis":"x","y":[0.6427,0.4852,0.4852,0.4852,0.4917,0.6427,0.6537,0.0,0.0,0.0,0.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"variable=F1<br>probability_threshold=%{x}<br>value=%{y}<extra></extra>","legendgroup":"F1","line":{"color":"#FFA15A","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"F1","orientation":"v","showlegend":true,"x":[0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0],"xaxis":"x","y":[0.4608,0.6533,0.6533,0.6533,0.6485,0.4608,0.445,0.0,0.0,0.0,0.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"variable=Kappa<br>probability_threshold=%{x}<br>value=%{y}<extra></extra>","legendgroup":"Kappa","line":{"color":"#19d3f3","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"Kappa","orientation":"v","showlegend":true,"x":[0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0],"xaxis":"x","y":[0.1732,0.0,0.0,0.0,0.024,0.1732,0.1713,0.0,0.0,0.0,0.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"variable=MCC<br>probability_threshold=%{x}<br>value=%{y}<extra></extra>","legendgroup":"MCC","line":{"color":"#FF6692","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"MCC","orientation":"v","showlegend":true,"x":[0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0],"xaxis":"x","y":[0.1923,0.0,0.0,0.0,0.0521,0.1923,0.1949,0.0,0.0,0.0,0.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"variable=LogLoss<br>probability_threshold=%{x}<br>value=%{y}<extra></extra>","legendgroup":"LogLoss","line":{"color":"#B6E880","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"LogLoss","orientation":"v","showlegend":true,"x":[0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0],"xaxis":"x","y":[14.0843,17.7821,17.7821,17.7821,17.2956,14.0843,14.0996,16.757,16.757,16.757,16.757],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"probability_threshold"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"value"}},"legend":{"title":{"text":"variable"},"tracegroupgap":0},"title":{"text":"Ada Boost Classifier Probability Threshold Optimization (default = 0.5)"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('3edf2bce-2a5a-4079-bd15-eadb047d8de1');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


    INFO:logs:returning model with best metric
    INFO:logs:optimize_threshold() succesfully completed......................................
    


```python
pred_holdout = pc.predict_model(calibrated_model)
```

    INFO:logs:Initializing predict_model()
    INFO:logs:predict_model(drift_kwargs=None, display=None, ml_usecase=MLUsecase.CLASSIFICATION, verbose=True, round=4, raw_score=False, drift_report=False, encoded_labels=False, probability_threshold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Preloading libraries
    INFO:logs:Preparing display monitor
    


<style  type="text/css" >
</style><table id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>        <th class="col_heading level0 col7" >MCC</th>        <th class="col_heading level0 col8" >LogLoss</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0row0_col0" class="data row0 col0" >Ada Boost Classifier</td>
                        <td id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0row0_col1" class="data row0 col1" >0.5886</td>
                        <td id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0row0_col2" class="data row0 col2" >0.5871</td>
                        <td id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0row0_col3" class="data row0 col3" >0.3418</td>
                        <td id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0row0_col4" class="data row0 col4" >0.5969</td>
                        <td id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0row0_col5" class="data row0 col5" >0.4347</td>
                        <td id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0row0_col6" class="data row0 col6" >0.1474</td>
                        <td id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0row0_col7" class="data row0 col7" >0.1616</td>
                        <td id="T_4b5670b4_c18d_11ec_8f13_00d76d400aa0row0_col8" class="data row0 col8" >14.2086</td>
            </tr>
    </tbody></table>



```python
pred_holdout
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lon</th>
      <th>shot_distance</th>
      <th>minutes_remaining_0</th>
      <th>minutes_remaining_1</th>
      <th>minutes_remaining_10</th>
      <th>minutes_remaining_11</th>
      <th>minutes_remaining_2</th>
      <th>minutes_remaining_3</th>
      <th>minutes_remaining_4</th>
      <th>...</th>
      <th>period_2</th>
      <th>period_3</th>
      <th>period_4</th>
      <th>period_5</th>
      <th>period_6</th>
      <th>period_7</th>
      <th>playoffs_1</th>
      <th>shot_made_flag</th>
      <th>Label</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33.950298</td>
      <td>-118.159798</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.5658</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34.027302</td>
      <td>-118.420799</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.5724</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.044300</td>
      <td>-118.269798</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.6437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33.981300</td>
      <td>-118.369797</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5663</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33.947300</td>
      <td>-118.284798</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4864</th>
      <td>33.964298</td>
      <td>-118.268799</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.5966</td>
    </tr>
    <tr>
      <th>4865</th>
      <td>33.886299</td>
      <td>-118.129799</td>
      <td>21.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.6176</td>
    </tr>
    <tr>
      <th>4866</th>
      <td>33.903301</td>
      <td>-118.381798</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5715</td>
    </tr>
    <tr>
      <th>4867</th>
      <td>33.921299</td>
      <td>-118.392799</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.5715</td>
    </tr>
    <tr>
      <th>4868</th>
      <td>33.896301</td>
      <td>-118.192802</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.5556</td>
    </tr>
  </tbody>
</table>
<p>4869 rows × 26 columns</p>
</div>




```python
def eval_metrics(actual, pred):
    return ({'Prec.': metrics.precision_score(actual, pred), 
             'Recall':metrics.recall_score(actual, pred),
             'F1':metrics.f1_score(actual, pred),
             'LogLoss':metrics.log_loss(actual, pred),
             'AUC':metrics.roc_auc_score(actual, pred),
             'Accuracy':metrics.accuracy_score(actual, pred),
             'Kappa':metrics.cohen_kappa_score(actual, pred),
             'MCC':metrics.matthews_corrcoef(actual, pred)})
            

```

Otimização dos hiperparâmetros

# 8 Registro do Modelo


```python
from mlflow.tracking import MlflowClient
import mlflow
import warnings
warnings.filterwarnings('ignore')
#mlflow.set_registry_uri("sqlite:///mlruns.db")
#from mlflow.models.signature import infer_signature
from sklearn import tree, preprocessing, metrics, model_selection
#from mlflow.models.signature import ModelSignature
mode_registre=calibrated_model
model_version = -1 
registered_model_name = 'Modelo Kobe Bryant'

with mlflow.start_run(experiment_id=experiment_id, run_name = 'RegistroModelo', nested=True):
    pred_holdout = pc.predict_model(calibrated_model)
    mr=metrics.precision_score(pred_holdout[target_col], pred_holdout['Label'])
    # Test set
    #pred_holdout = pc.predict_model(model_to_registre)
    #pr = metrics.precision_score(pred_holdout[target_col], pred_holdout['Label'])
    #if pr > min_precision:
       # print(f'=> Aceito o modelo com precisão {pr} (min: {min_precision})')
        # Pycaret exporta junto o pipeline de preprocessamento
    pc.save_model(mode_registre, f'./{registered_model_name}') 
        # Carrega novamente o pipeline + bestmodel
    model_pipe = pc.load_model(f'./{registered_model_name}')
        # Assinatura do Modelo Inferida pelo MLFlow
    model_features = list(df_kb_tt.drop(target_col, axis=1).columns)
        #inf_signature = infer_signature(DataBin[model_features], model_pipe.predict(DataBin))
        # Exemplo de entrada para o MLmodel
        #input_example = {x: DataBin[x].values[:nexamples] for x in model_features}
        # Log do pipeline de modelagem do sklearn e registrar como uma nova versao
    mlflow.sklearn.log_model(
        sk_model=model_pipe,
        artifact_path="sklearn-model",
        registered_model_name=registered_model_name,
            #signature = inf_signature,
            #input_example = input_example
        )
        # Criacao do cliente do servico MLFlow e atualizacao versao modelo
    client = MlflowClient()
    if model_version == -1:
        model_version = client.get_latest_versions(registered_model_name)[-1].version
        # Registrar o modelo como staging
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version, # Verificar com usuario qual versao
        stage="Staging"
    )
    result= eval_metrics(pred_holdout[target_col].values, pred_holdout['Label'].values)
    result_title=''
    result_value=''
    for metric in result.keys():    
        mlflow.log_metric(metric, result[metric])
        print('{:<8}\t{:0.2f}'.format(metric, result[metric]))

    mlflow.log_metric('Version', model_version)
#else:
    #print(f'=> Rejeitado o modelo com precisão {pr} (min: {min_precision})')

mlflow.end_run()

```

    INFO:logs:Initializing predict_model()
    INFO:logs:predict_model(drift_kwargs=None, display=None, ml_usecase=MLUsecase.CLASSIFICATION, verbose=True, round=4, raw_score=False, drift_report=False, encoded_labels=False, probability_threshold=None, estimator=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Checking exceptions
    INFO:logs:Preloading libraries
    INFO:logs:Preparing display monitor
    


<style  type="text/css" >
</style><table id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>        <th class="col_heading level0 col7" >MCC</th>        <th class="col_heading level0 col8" >LogLoss</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0row0_col0" class="data row0 col0" >Ada Boost Classifier</td>
                        <td id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0row0_col1" class="data row0 col1" >0.5886</td>
                        <td id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0row0_col2" class="data row0 col2" >0.5871</td>
                        <td id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0row0_col3" class="data row0 col3" >0.3418</td>
                        <td id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0row0_col4" class="data row0 col4" >0.5969</td>
                        <td id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0row0_col5" class="data row0 col5" >0.4347</td>
                        <td id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0row0_col6" class="data row0 col6" >0.1474</td>
                        <td id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0row0_col7" class="data row0 col7" >0.1616</td>
                        <td id="T_4c4d1b46_c18d_11ec_a36f_00d76d400aa0row0_col8" class="data row0 col8" >14.2086</td>
            </tr>
    </tbody></table>


    INFO:logs:Initializing save_model()
    INFO:logs:save_model(kwargs={}, verbose=True, prep_pipe_=Pipeline(memory=None,
             steps=[('dtypes',
                     DataTypes_Auto_infer(categorical_features=[],
                                          display_types=False, features_todrop=[],
                                          id_columns=[],
                                          ml_usecase='classification',
                                          numerical_features=[],
                                          target='shot_made_flag',
                                          time_features=[])),
                    ('imputer',
                     Simple_Imputer(categorical_strategy='not_available',
                                    fill_value_categorical=None,
                                    fill_value_numerical=None,
                                    nume...
                    ('scaling', 'passthrough'), ('P_transform', 'passthrough'),
                    ('binn', 'passthrough'), ('rem_outliers', 'passthrough'),
                    ('cluster_all', 'passthrough'),
                    ('dummy', Dummify(target='shot_made_flag')),
                    ('fix_perfect', Remove_100(target='shot_made_flag')),
                    ('clean_names', Clean_Colum_Names()),
                    ('feature_select', 'passthrough'), ('fix_multi', 'passthrough'),
                    ('dfs', 'passthrough'), ('pca', 'passthrough')],
             verbose=False), model_name=./Modelo Kobe Bryant, model=CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                             base_estimator=None,
                                                             learning_rate=0.4,
                                                             n_estimators=280,
                                                             random_state=4678),
                           cv=5, method='sigmoid'))
    INFO:logs:Adding model into prep_pipe
    INFO:logs:./Modelo Kobe Bryant.pkl saved in current working directory
    INFO:logs:Pipeline(memory=None,
             steps=[('dtypes',
                     DataTypes_Auto_infer(categorical_features=[],
                                          display_types=False, features_todrop=[],
                                          id_columns=[],
                                          ml_usecase='classification',
                                          numerical_features=[],
                                          target='shot_made_flag',
                                          time_features=[])),
                    ('imputer',
                     Simple_Imputer(categorical_strategy='not_available',
                                    fill_value_categorical=None,
                                    fill_value_numerical=None,
                                    nume...
                    ('fix_perfect', Remove_100(target='shot_made_flag')),
                    ('clean_names', Clean_Colum_Names()),
                    ('feature_select', 'passthrough'), ('fix_multi', 'passthrough'),
                    ('dfs', 'passthrough'), ('pca', 'passthrough'),
                    ['trained_model',
                     CalibratedClassifierCV(base_estimator=AdaBoostClassifier(algorithm='SAMME',
                                                                              base_estimator=None,
                                                                              learning_rate=0.4,
                                                                              n_estimators=280,
                                                                              random_state=4678),
                                            cv=5, method='sigmoid')]],
             verbose=False)
    INFO:logs:save_model() successfully completed......................................
    INFO:logs:Initializing load_model()
    INFO:logs:load_model(verbose=True, authentication=None, platform=None, model_name=./Modelo Kobe Bryant)
    

    Transformation Pipeline and Model Successfully Saved
    Transformation Pipeline and Model Successfully Loaded
    

    Registered model 'Modelo Kobe Bryant' already exists. Creating a new version of this model...
    2022/04/21 12:08:41 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation.                     Model name: Modelo Kobe Bryant, version 11
    

    Prec.   	0.60
    Recall  	0.34
    F1      	0.43
    LogLoss 	14.21
    AUC     	0.57
    Accuracy	0.59
    Kappa   	0.15
    MCC     	0.16
    

    Created version '11' of model 'Modelo Kobe Bryant'.
    

Ativando o serviço Server para o modelo Modelo Kobe Bryant em Staging, execução em outro notebook


```python
#import os
#os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlruns.db'

#!mlflow models serve -m "models:/modelo_cancer/Staging" --no-conda -p 5001
```

# 8.1 Revalidação

Para a revalidação será feito utilizando os dados com 3PT Field Goal, que são diferentes porque o acerremeço de 3 pontos é mais
distânte do de 2 pontos. Dessa forma representa um novo conjunto de dados com caracteristicas, digamos que não esperadas pelo modelo,
que foi treinado com dados de arremeços de 2pts.

Abaixo esta uma função para calculo das principais metricas e retorno em formato dicionário.

O Serviço vai enviar uma request http para o serviço da API que realiza a predição e retorna os valores preditos em um JSON que é 
convertido para DataFrame e então são calculadas as metricas. Tudas as metricas são então salvas como log metric no MLFLow


```python
import pandas as pd
import requests
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc


#Configuração do request
host = 'localhost'
port = '5001'
url = f'http://{host}:{port}/invocations'
headers = {'Content-Type': 'application/json',}

with mlflow.start_run(experiment_id=experiment_id, run_name = 'RevalidaçãoModelo', nested=True):
    #Dados para revalidação
    df_kb_op=pd.read_parquet('../Data/Operalization/base_operation.parquet')
    http_data = df_kb_op.drop(target_col,axis=1).to_json(orient='split')
    r = requests.post(url=url, headers=headers, data=http_data)
    df_kb_op.loc[:, 'operation_label'] = pd.read_json(r.text).values[:,0]
    df_kb_op.to_parquet('../Data/Operalization/base_operation_processed.parquet')
    #ll = log_loss(df_kb_op[target_col], df_kb_op['operation_label'])
    #f1 = f1_score(df_kb_op[target_col], df_kb_op['operation_label'])
    #acc= accuracy_score(df_kb_op[target_col], df_kb_op['operation_label'])
    #auc=auc(df_kb_op[target_col], df_kb_op['operation_label'])
    result= eval_metrics(df_kb_op[target_col], df_kb_op['operation_label'])
    result_title=''
    result_value=''
    for metric in result.keys():    
        mlflow.log_metric(metric, result[metric])
        print('{:<8}\t{:0.2f}'.format(metric, result[metric]))

    mlflow.log_metric('Version', model_version)
mlflow.end_run()

```

    Prec.   	0.63
    Recall  	0.39
    F1      	0.48
    LogLoss 	13.74
    AUC     	0.59
    Accuracy	0.60
    Kappa   	0.19
    MCC     	0.20
    


```python

```

Comparação

# 8.a Aderência com Novo Conjunto de Dados


```python
df_ex = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])
df_ex_fh = df_ex[df_ex['status'] == 'FINISHED'].copy()
df_ex_fh_rv = df_ex_fh[df_ex['tags.mlflow.runName'] == 'RevalidaçãoModelo'].copy()
df_ex_fh_rg = df_ex_fh[df_ex['tags.mlflow.runName'] == 'Gradient Boosting Classifier'].copy()
metrics_select = ['tags.mlflow.runName','metrics.LogLoss', 'metrics.F1','metrics.Accuracy', 'metrics.Prec.', 'metrics.Recall']
df_ex_fh_rv_fl=df_ex_fh_rv[metrics_select].copy()
df_ex_fh_rg_fl=df_ex_fh_rg[metrics_select].copy()
#print(df_ex_fh_rv_fl.keys())
df_rs=pd.concat([pd.DataFrame(df_ex_fh_rv_fl.iloc[:1]), df_ex_fh_rg_fl.iloc[:1]], axis=0)
df_rs.to_parquet('../Data/Operalization/results/results01.parquet')
df_rs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tags.mlflow.runName</th>
      <th>metrics.LogLoss</th>
      <th>metrics.F1</th>
      <th>metrics.Accuracy</th>
      <th>metrics.Prec.</th>
      <th>metrics.Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RevalidaçãoModelo</td>
      <td>13.740678</td>
      <td>0.478008</td>
      <td>0.602169</td>
      <td>0.630009</td>
      <td>0.385096</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Gradient Boosting Classifier</td>
      <td>14.005300</td>
      <td>0.461300</td>
      <td>0.594500</td>
      <td>0.626600</td>
      <td>0.365200</td>
    </tr>
  </tbody>
</table>
</div>



# 8b Monitoramento do Modelo

Na comparação entre os resultados obtidos no experimênto de Ada Boost Classifier que foi o melhor modelo escolhido pelo Pycaret
e o RelalidaçãoModelo que foi realizado com os dados de arremeços de 3 pontos. As metricas ficaram muito próximas, significa que 
não houve perca de performace, com resultados até mesmo com metricas acima dos resultados coletados pelo algoritmo de melhor resultado. O correto séria com 
esses dados novos, realizar uma nova amostra de dados de forma aleatoria e estratificada, unindo os arremeços de 2 e de 3 para gerar uma melhora nos resultados de operação, com intuito de manter as matricas que perderam cairam mais como a Prec. e o Recall e o F1 mais alinhados com os resultados do desenvolvimento. 

# 8c Estrategias Reativa e Preditiva

O monitoramnto do modelo pode ser feito com a variável resposta, como no caso acima, onde foi comparada a operação com os dados 
do desenvolvimento para gerar indicadores, e os indicadores servem para descrever como está a saúde do modelo. Ou no caso de não existir
a variável resposta, deve ser utilizada uma forma de gerar a variável, seja com equipe de especialista na área em questão, ou utilizando
outros algoritmos por exemplo. Mas nesse caso específico a variável resposta existe porque cada arremeço gera uma cesta ou erro.
Num caso por exemplo de tratamento de uma doença, ou ainda  não se saiba, pode ser coletada amostras tem realização de exames onde 
e apoio de especialistas, onde se possa gerar os resultados, e daqui extrair as metricas.

# 9 Streamlit

Para o acompanhamento da operação do modelo, está sendo utilizado visualizações graficas no streamlit. Os graficos então sendo
construídos utilizando informações do MLFLow do experimento e também dos artefatos gerados no experimento. E o streamlit fazendo um 
papel de front-end para exibição.

A navegação está no slidebar no lado esquerdo da tela, uma selectbox com opções para selecionar escolhe a visualização, Inicial, 
Versionamento, Operação. No Versionamento, os gráficos que comparam dada uma versão de Staging, a operação e o registro, ou seja,
as metricas do teste com algoritmo durante o registro para Staging, e o algoritmo durante a operação simulada em Staging. Caso houver duas simulações por exemplo, com a mesma versão (operação) há dois ou mais pontos para com as metricas. A opção visualização escolhe o gráfico, como LogLoss ou F1-Score.

# Versionamento

Um Gráfico mostra a versão do modelo como eixo X e a métrica como eixo Y, Uma linha é o registro e outra a simulação da operação


![VLL.png](attachment:VLL.png)

E que mostra que os resultados de operação foram melhores que os de treino e test em todas as versões

Da mesma forma o gráfico de F1-Score

![Vf1.png](attachment:Vf1.png)


```python
Outros graficos só de operação como curva_roc
```

![Oroc.png](attachment:Oroc.png)

E um com a Precisão, Acurácia e Recall nas validações da operação por versão do modelo registrada no MLFlow. Na verão 8, houveram
duas operações de revalidação, por isso ela aparece duas vezes.


![OAcc.png](attachment:OAcc.png)

No fim os resultados das metricas não foram bons, possívelmente, pode ser fruto da seleção de metricas inicial que foi realizada. 
Pode ser que que uma boa opção seria avaliar as melhores metricas antes da seleção. Porém o objetivo principal do trabalho é demonstrar
o processo de Auto ML com a utilização das ferramentas Pycaret, MLFlow, Streamlit, Sklearn, principalmente. Principalmente por que também
há, Jupyter notebook, Ambiente Anaconda, entre outros. 

Este processo é ciclico, então, haveria novos preprocessamentos, com novo treino e teste, registro e versão e monitoramento ou simulação da operação. Os gráficos teriam a verão 12 incluída, e assim por diante.


```python

```
