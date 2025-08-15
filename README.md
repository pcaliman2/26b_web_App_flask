# PROYECTO FLASK

La version funcional de Este proyecto se esta ejecutando en el siguiente URL
https://26bwebappflask-production.up.railway.app/

La idea detras de este proyecto es crear un calculo del MOS WCDMA en un determinado punto.   El MOS (Mean Opinion Score) es una medida de percepción de la calidad de la Red que experimenta un usuario.   El MOS el un valor que tiene una escalad de 1 a 5 donde 1 es muy malo y 5 muy bueno. 

## Structure

The project is organized as follows:

- **`src/app.py`** → Es la aplicació que ejecuta el servicio del server de flask.
- **`src/Creacion_Modelo.ipynb`** → Notebook que contiene la creación del modelo del MOS.
- **`src/run_waitress.py`** → Es la programa que inicia el servicio de flask en el navegador, ejecuta a app.py y relaciona el url al puerto 5000.


## Ejecutar la Aplicacion

Para ejecutar la aplicación se debe correr el servicio de flask a trave del archivo run_waitress.py que esta en el directorio src:

python /src/run_waitress.py
## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.
