import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Cargamos los datos desde seaborn
sns.get_dataset_names()
diamonds=sns.load_dataset('diamonds')

st.title('Análisis de datos diamonds')
col1, col2 = st.columns([3, 1])


plot=sns.scatterplot(x='carat', y='price', data=diamonds)

#Presentamos los datos en la app de streamlit


#Mostramos las columnas disponibles para el filtro

# st.write(diamonds.head())
col1.pyplot(plot.get_figure())
col2.write(diamonds.head())

#Mostramos la gráfica en la app
# st.pyplot(plot.get_figure())

X=diamonds[['carat']]
y=diamonds['price']

model_RL=LinearRegression()

#Entrenamos el modelo

model_RL.fit(X,y)

#Mostramos los coeficientes del modelo

st.write('Coeficiente: ', model_RL.coef_)
st.write('Intercepto: ', model_RL.intercept_)

#Pedimos al usuario que introduzca un valor para el carat

carat_input=st.number_input('Introduce un valor para el carat:')

#Predecimos el precio del diamante

prediction=model_RL.predict([[carat_input]])

st.write('El precio del diamante con', carat_input, 'carat es:', prediction)

#Dividimos los datos en entrenamiento y prueba

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenamos un nuevo modelo con los datos de entrenamiento

model_RL_train=LinearRegression()
model_RL_train.fit(X_train, y_train)

#Calculamos el error cuadrático medio

error_cuadratico_medio=(model_RL_train.predict(X_test)-y_test)**2

st.write('Error cuadrático medio:', error_cuadratico_medio.mean())

#Calculamos el coeficiente de determinación

r_cuadrado=1-(error_cuadratico_medio.sum()/(y_test-y_test.mean())**2)

st.write('Coeficiente de determinación:', r_cuadrado)
#y_train_a=np.reshape(y_train,(-1,1))
#Mostramos la gráfica de la regresión lineal

#plot_train=sns.scatterplot(x=X_train, y=y_train, color='blue')
#plot_train=sns.lineplot(x=X_test, y=model_RL_train.predict(X_test), color='red')
plot_train=plt.scatter(x=X_train, y=y_train)
plot_train2=plt.plot(x=X_test, y=model_RL_train.predict(X_test),color='green')
st.pyplot(plot_train.get_figure())
st.pyplot(plot_train2.get_figure())
