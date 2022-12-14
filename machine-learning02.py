import tensorflow as tf 
import pandas as pd
import numpy as np
# librerias de graficos
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # nos permite dividir un dataset en dos bloques, entrenamiento y validación del modelo
from sklearn.linear_model import LinearRegression # predice el valor de datos desconocidos mediante el uso de otro valor de datos relacionado y conocido
from sklearn.metrics import r2_score # Función de puntuación de regresión 

# Cargar DataSet
ventas_df= pd.read_csv("/content/IceCreamData.csv")

#Visualziar los datos
###sns.scatterplot(ventas_df['Temperature'],ventas_df['Revenue'], )


#cargando los datos en el set
x_train = ventas_df['Temperature']
y_train = ventas_df['Revenue']

#Crear Modelo
model = tf.keras.Sequential() #modelo vacio
#pongo en comentario para trabajar con mas capas y neuronas 
#model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#Trabajando con mas capas
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
oculta3 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
#Creando el modelo en base a las capas trabajadas
model = tf.keras.Sequential([oculta1, oculta2, oculta3, salida])

#Imprimir summary modelo
#model.summary() #poner en comentario ya que es solo visualizar

#compilando ingresar peso y sesgo
# mean_squared_error
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss= 'mean_squared_error')

#entrenamiento
epochs_hits = model.fit(x_train, y_train, epochs = 500, verbose=False)

#evaluar
keys = epochs_hits.history.keys()
#print(keys)

#Grafico
plt.plot(epochs_hits.history['loss'])
plt.title('Progreso de perdida durante entrenamiento')
plt.xlabel('# Epochs')
plt.ylabel('Magnitud de perdidas (loss)')
plt.legend('Entrenamiento del Modelo')

weights = model.get_weights()
###print(weights)

##medir cuanto dinero gano o pierdo teniendo en cuenta la temperatura
Temperatura = 30
Revenue = model.predict([Temperatura])
print(f"La ganancia según la red neuronal es de {Revenue}")

#Grafico
# la prediccion de datos 
plt.scatter(x_train, y_train, color= 'red')
plt.plot(x_train,model.predict([x_train]), color= 'gray' )
plt.title('Ganancia Generada vs el Clima en clase de IA')
plt.xlabel('Temperatura Celsius')
plt.ylabel('Ganancia RD$')
plt.legend('Entrenamiento del Modelo')

#MEDIR EL NIVEL DE PRECISIÓN

data = ventas_df.values
x, y = data[:,:-1], data[:,-1] #El conjunto de datos contendrá las entradas en la matriz bidimensional x y las salidas en la matriz unidimensional y
#[:] significa todas las filas
#[ : , -1] significa el último elemento en todas las filas
#[ : , :-1 ] significa todas las filas con todos los elementos en filas excepto el último

#print(x)
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)  # splitting in the ratio 80:20 
# train_test_split permite dividir los datos en conjuntos de entrenamiento y conjuntos de prueba
# test_size es el número que define el tamaño del conjunto de prueba.
# random_state es el objeto que controla la aleatorización durante la división.

model = LinearRegression()

model.fit(x_train, y_train)

#Making Predictions and Checking Accuracy
#Aquí se predice a partir del valor de la temperatura
y_pred = model.predict(X_test)

# Aquí calculamos el nivel de relación entre variables (regresión)
accurancy = round((r2_score(y_test, y_pred)),4) *100
#valores correctos y valores estimados o predictivos.

print(f"La precisión de este modelo es de: {accurancy} %")

