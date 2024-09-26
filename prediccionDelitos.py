import streamlit as st

#importar las bibliotecas tradicionales de numpy y pandas
import numpy as np
import pandas as pd

#importar las biliotecas graficas e imágenes
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt


import joblib as jb

from sklearn.ensemble import RandomForestClassifier



imagen_video = Image.open("delitos-federales.jpg") 


#Librerias no usadas
#from streamlit_lottie import st_lottie
#import requests

## Iniciar barra lateral en la página web y título e ícono

st.set_page_config(
  page_title="ML Delitos Bucaramanga",
  page_icon="4321369.ico",
  initial_sidebar_state='auto'
  )

@st.cache_data
def load_data():
  df= pd.read_csv('Delito Bucaramanga_preprocesar.csv', delimiter=",") #Currently on my local machine
  return df
df= load_data()

@st.cache_resource
def load_models():
  codEdad=jb.load('codEdad.bin')
  codHorario=jb.load('codHorario.bin')
  codGenero=jb.load('codGenero.bin')
  codDia=jb.load('codDia.bin')
  codComuna=jb.load('codComuna.bin')
  modeloBA=jb.load('modeloBA.bin')
  return codEdad,codHorario,codGenero,codDia,codComuna,modeloBA
codEdad,codHorario,codGenero,codDia,codComuna,modeloBA = load_models()

#Primer contenedor
st.markdown('----')
with st.container():
  st.subheader("Modelo Machine Learning para prevencion de delitos en Bucaramanga")
  st.title("Reporte Final ")
  st.write("Realizado por Alfredo Díaz Claros:wave:")
  st.write("""

**Introducción** 
Los datos fueron tomados con la Información de los delitos ocurridos en el municipio de Bucaramanfa,según la modalidad y 
conducta delictiva, barrios y comunas de ocurrencia, armas y medios empleados, móvil del agresor y de la víctima,
curso de vida y género de la víctima, con una desagregación temporal por mes, día, y hora de ocurrencia.

Datos Actualizados en la fuente: 2 de septiembre de 2023

Esta aplicacion es el resultado de un modelo de “machine learning” para predecir el delito en Bucaramanga, 
ciudad intermedia de Colombia. 
Se utilizó modelos supervidados de clasificacion y de obtuvo los mejores resutlado con
Se identificó que los mejores resultados en la predicción del crimen se dieron con RandomForestClassifier-

A pesar de que existen limitaciones con la información útil para la predicción, se probarion 6 opciones y el mejor modelp
arrojó  más del 66 % de exactitud (accuracy.).
Concluimo que los modelos de predicción del delito basado en los articulos del código civil
que se constituyen una herramienta útil para construir estrategias de prevención pero el objetivo no es estigmatizar
los barrios o comunas de la ciudad.
""")


st.markdown('----')
with st.container():
  st.write("---")
  left_column, right_column = st.columns(2)
  with left_column:
    st.subheader("Las lbrerias usadas para entrenar el demos fueron")
    st.write(
      """
      El objetivo de este trabajo acadeémico es construir una herramienta en código Python para predecir la categoria
      de delito que se comete tienen en cuentas las siguietes caracterísitas:
      'EDAD', 'GENERO', 'RANGO_HORARIO_ORDEN', 'NOM_COM', 'DIA_NOMBRE'.
      El modelo elegido fue Bosque Aleatorio.
      
      Este proyecto ayuda que con la información las personas y las autoridades definan acciones eficientes
      en la prevención del delito concentrando sus recursos en pequeñas unidades geográficas, duplicar los tiempos de patrullaje en zonas 
      combinando tiempos adicionales de patrullaje y en general los ciudadanos tomar sus medidas preventivas.
      Al frente se encuenta el  codigo finalmente usado despúes de la etapa de ingeniería de caracteristicas.
      """
    )

  with right_column:
      st.subheader("Código")
      code = '''
      # TRATAMIENTO DE DATOS
      
      import pandas as pd
      import numpy as np
      
      SISTEMA OPERATIVO
      
      import os
      
      # GRAFICO
      import matplotlib.pyplot as plt
      import matplotlib.ticker as ticker
      import seaborn as sns
      import urllib
      from sklearn.metrics import confusion_matrix
      from sklearn.metrics import ConfusionMatrixDisplay
      from sklearn.metrics import classification_report
      from sklearn.metrics import accuracy_score
      from sklearn.feature_selection import SelectKBest
      from sklearn.feature_selection import f_classif
      from sklearn.ensemble import RandomForestClassifier
      
      Defino el algoritmo a utilizar
      modeloBA= RandomForestClassifier(random_state=0)
      
      Entreno el modelo
      modeloBA.fit(X_train, y_train)
      
      accuracy del set de entrenamiento
      modeloBA.score(X_train,y_train)*100
      modeloBA.score(X_test,y_test)*100
     '''
      st.code(code, language="python", line_numbers=True)
      
edades=['ADOLECENCIA','ADULTEZ','INFANCIA','JUVENTUD','PERSONA MAYOR','PRIMERA INFANCIA']
horas=['MADRUGADA','MAÑANA','NOCHE','TARDE']
comunas=['CABECERA DEL LLANO','CENTRO', 'GARCIA ROVIRA', 'LA CIUDADELA',
 'LA CONCORDIA', 'LA PEDREGOSA', 'LAGOS DEL CACIQUE', 'MORRORICO', 'MUTIS',
 'NORORIENTAL', 'NORTE', 'OCCIDENTAL', 'ORIENTAL', 'PROVENZA', 'SAN FRANCISCO',
 'SUR', 'SUROCCIDENTE']
generos=['FEMENINO','MASCULINO']
diaSemana=['lunes','martes','miércoles','jueves','sábado','viernes','domingo']

st.subheader("Detalle del dataset usado en el proyecto")

st.write("El número de registros cargados es: ", len(df))
#st.write("comprendido desde ", pd.to_datetime(df['FECHA_COMPLETA']).min(), " hasta ", pd.to_datetime(df['FECHA_COMPLETA']).max())
st.write("El númerp de tipos de delitos registrados  ", len(df['DELITO_SOLO'].unique()), ", de", len(df['BARRIOS_HECHO'].unique()), "barrios en ",len(df['NOM_COM'].unique()),  " comunas")
st.write(df.head(5))
#Opciones de la barra lateral

logo=Image.open("menu.jpg")
st.sidebar.write('...')
st.sidebar.image(logo, width=100)
st.sidebar.header('Seleccione los datos de entrada')


def seleccionar(generos,comunas, diaSemana,edades,horas):

  #Filtrar por municipio

  st.sidebar.subheader('Selector del Género')
  genero=st.sidebar.selectbox("Seleccione el genero",generos)

  #Filtrar por estaciones
  st.sidebar.subheader('Selector del dia de la semana')
  dia=st.sidebar.selectbox("Selecciones del dia de la semana",diaSemana)
  
  #Filtrar por estaciones
  st.sidebar.subheader('Selector del dia del rango de edad')
  edad=st.sidebar.selectbox("Selecciones la edad",edades)
  
  #Filtrar por estaciones
  st.sidebar.subheader('Selector del rengo de dia')
  hora=st.sidebar.selectbox("Seleccione la jornada ",horas)
  
  st.sidebar.subheader('Selector de mes') 
  mes=st.sidebar.slider('número del mes', 1, 12, 1)
  
  #Filtrar por departamento
  st.sidebar.subheader('Selector de comuna')
  comuna=st.sidebar.selectbox("Seleccione la comuna",comunas)

  
  return edad,genero,mes,hora,comuna,dia

edad,genero,mes,hora,comuna,dia=seleccionar(generos,comunas,diaSemana,edades,horas)



#st.write(datos.describe())
with st.container():
  st.subheader("Predición")
  st.title("Predicción de Articulo del Código Civil Colombiano")
  st.write("""
           El siguiente es el pronóstico de la clase delito usando el modelo usando los diferentes umbrales
           """)
           
  edadn=list(codEdad.transform([edad]))[0]
  horan=list(codHorario.transform([hora]))[0]
  dian=list(codDia.transform([dia]))[0]
  comunan=list(codComuna.transform([comuna]))[0]
  generon=list(codGenero.transform([genero]))[0]
  lista=[[generon,mes,comunan,dian,edadn,horan]]
  
  st.write("Se han seleccionado los siguientes parámetros:")
  st.write("Edad: ", edad, "equvalente a",edadn )
  st.write("Género : ", genero,"equvalente a",generon)
  st.write("Mes :", mes,"equvalente a",mes)
  st.write("Hora", hora,"equvalente a", horan)
  st.write("Comuna",comuna,"equvalente a", comunan)
  st.write("dia",dia,"equvalente a",dian) 
  
  X_predecir=pd.DataFrame(lista,columns=['GENERO','MES_NUM','NOM_COM','DIA_NOMBRE','RangoEdad','rangoHORARIO'])
  y_predict=modeloBA.predict(X_predecir)
  st.markdown('----')
  st.title(':blue[La predicción es:]' )
  st.title(y_predict[0])
  st.markdown('----')
  dfc=df[(df["GENERO"]==genero) & (df["MES_NUM"]==mes) & (df["NOM_COM"]==comuna) & (df["DIA_NOMBRE"]==dia) & (df["TIPOLOGÍA"]==y_predict[0])]
  solo=dfc['DELITO_SOLO'].value_counts()/dfc['DELITO_SOLO'].size
  solo.rename({'count':'Frecuencia'}, inplace = True)
  barrios=dfc['BARRIOS_HECHO'].value_counts()/dfc['BARRIOS_HECHO'].size
  barrios.rename({'count':'Frecuencia'}, inplace = True)
  
st.markdown('----')    
with st.container():
  if len(solo)!=0:
    st.subheader("Análisis gráficos")
    st.subheader("Delitos similares cometidos con los parámetros dados")
    st.write("""
           Para apoyar el análisis y la toma de decisiones se presentan los delites cometidos en esas
           opciones.
           """)

    st.write(dfc[['BARRIOS_HECHO','DESCRIPCION_CONDUCTA', 'ARMAS_MEDIOS',
        'MOVIL_VICTIMA','DELITO_SOLO', 'MOVIL_AGRESOR', 'CLASE_SITIO']])
    
    st.write('Frecuencia en los barrios de la comuna  '+ comuna + '  es: ', solo )
    
    st.write('Frecuencia en los barrios de la comuna  '+ comuna + '  es :',barrios )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(barrios,labels=barrios.index, autopct='%1.1f%%')
    st.pyplot(fig)
