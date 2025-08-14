import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


#celcius = np.array ([-40, -10,0,8,15,22,38], dtype = float)
#farenh = np.array ([-40,18, 33,46,59,62,200], dtype = float)

#celcius = np.array ([-40, -10,0,8,15,22,38], dtype = float)
#farenh = np.array ([-40,14, 32,46.4,59,71.6,100.4], dtype = float)



#selected_measurements = ['wcdma_aset_ecio_avg', 'wcdma_aset_rscp_avg',  'wcdma_txagc', 'wcdma_bler_average_percent_all_channels', 'wcdma_rssi','gsm_speechcodecrx']


mos_csv = pd.read_csv("C:\MOS_Islas\wcdma_DataEntre.csv")
selected_measurements = ['wcdma_aset_ecio_avg', 'wcdma_aset_rscp_avg', 'wcdma_txagc', 'wcdma_bler_average_percent_all_channels','gsm_speechcodecrx']
selected_output = [ 'mos']
#estos son los Valore del Entrenamiento'
datos_entrada = mos_csv[selected_measurements]
mos_output = mos_csv[selected_output]



#Aqui va el csv que tiene lo datos a ser calculado'
estudio_csv = pd.read_csv("C:\MOS_Islas\Faltante_02.csv")
datos_prueba = estudio_csv[selected_measurements]




#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#x_scaled = scaler.fit_transform(datos_entrada)


entrada = tf.keras.layers.Dense(units=5, input_shape=[5])
oculto1 = tf.keras.layers.Dense(units=8)
oculto2 = tf.keras.layers.Dense(units=8)
oculto3 = tf.keras.layers.Dense(units=8)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([entrada, oculto1, oculto2, oculto3, salida])

modelo.compile(
  optimizer=tf.keras.optimizers.Adam(0.001),
  loss='mean_squared_error'
)

print("Comenzando Entrenamiento...")
historial = modelo.fit(datos_entrada , mos_output, epochs =50 , verbose = True)
#historial = modelo.fit(celcius, farenh, epochs = 1000)
print('Modelo entrenado')


#import matplotlib.pyplot as plt
#plt.xlabel('# Epoca')
#plt.ylabel('magnitud de perdida')
#plt.plot(historial.history['loss'])



aa=datos_prueba
f = open ('C:\MOS_Islas\SalidaMOS_Temp2.txt','w')

a = np.array(datos_prueba)
for j in range(len(a)):
    a0=a[j,0]
    a1=a[j,1]
    a2=a[j,2]
    a3=a[j,3]
    a4=a[j,4]
    aa=[[a0,a1,a2,a3,a4]]
    resultado = modelo.predict(aa)
    f.write(str(resultado)) 
    f.write("\n")
    print(j)
    #print(str(resultado))
f.close()

print('Concluido')






    
    
    
    
    
    