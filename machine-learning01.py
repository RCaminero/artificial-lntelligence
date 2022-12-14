import pandas as pd
df = pd.read_csv("/content/calificaciones.csv")


import tensorflow as tf 
import pandas as pd
import numpy as np
# librerias de graficos
import seaborn as sns
import matplotlib.pyplot as plt

#Importar los datos desde el archivo
df = pd.read_csv("/content/datos.csv")

#Visualizacion
#sns.scatterplot(df['celsius'], df['fahrenheit'])

#cargando los datos en el set
x_train = df['celsius']
y_train = df['fahrenheit']

#Crear Modelo
model = tf.keras.Sequential() #modelo vacio
#pongo en comentario para trabajar con mas capas y neuronas 
#model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#Trabajando con mas capas
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
#Creando el modelo en base a las capaas trabajadas
model = tf.keras.Sequential([oculta1, oculta2, salida])

#Imprimir summary modelo
#model.summary() #poner en comentario ya que es solo visualizar

#compilando ingresar peso y sesgo
# mean_squared_error
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss= 'mean_squared_error')

#entrenar modelo
epochs_hits = model.fit(x_train , y_train, epochs= 30, verbose = False)

# Evaluar
epochs_hits.history.keys()

#Grafico
plt.plot(epochs_hits.history['loss'])
plt.title('Progreso de perdida durante entrenamiento')
plt.xlabel('# epochs')
plt.ylabel('Magnitud de perdidas (loss)')
plt.legend('Entrenamiento del Modelo')

model.get_weights()

#Prediccciones
vTempC = 120
vTempF = (vTempC * (9/5)) + 32
print(f"Formula de Conversion :  {vTempF}")

#Trabajando con el modelo
vTempF = model.predict([vTempC])
print(f"Modelo Predictivo :  {vTempF}")


