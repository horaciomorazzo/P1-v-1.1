
Horacio Morazzo

Proyecto Final Individual 1 - Data Science

Introducción

El siguiente proyecto demuestra la factibilidad de desarrollar un sistema de recomendación de juegos para la plataforma Steam.
Dicho sistema deberá ser accesible desde un navegador web.
Contamos con 3 datasets, provistos por Steam, con información relacionada a los juegos para alimentar nuestro sistema de recomendación.

Requisitos solicitados:

Se deben crear las siguientes funciones para los endpoints que se consumirán en la API

def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def UsersWorstDeveloper( año : int ): Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def sentiment_analysis( empresa desarrolladora : str ): Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor.
Ejemplo de retorno: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}

Machine learning

Sistema de recomendación item-item

Se deberá crear la siguiente función:

def recomendacion_juego( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Para este sistema se utilizará la técnica de Similitud del Coseno. La similitud del coseno es la medida de similitud entre dos vectores, calculando el coseno del ángulo entre dos vectores proyectados en un espacio multidimensional. Se puede aplicar a elementos disponibles en un conjunto de datos para calcular la similitud entre sí mediante palabras clave u otras métricas.

Desarrollo

Inicialmente se trabaja en el ETL de los dataset suministrados. Este punto del proceso fue el que demandó la mayor cantidad de tiempo. El formato utilizado por los dataset (json) no resultó demasiado sencillo de leer, contenía gran cantidad de datos faltantes, problemas de tipos, etc. 
Una vez limpios y leídos, se guardaron inicialmente en formato CSV.

Luego se procedió a diseñar las funciones solicitadas. El proceso de elaboración del código de cada una de ellas compartió, en líneas generales, la misma técnica. Armar un dataset utilizando las columnas necesarias de cada uno de los 3 suministrados y luego agrupar y filtrar la información, acorde a lo requerido. Finalmente se devolvía el resultado con el formato adecuado.
El código se encuentra comentado línea por línea en su inmensa mayoría, por lo cual se puede revisar para mayor detalle.

Una vez obtenidas las funciones funcionando correctamente, se incorporaron a FastAPI.

Para acceder a la API desde la web, se utilizó una versión gratuita de Render con una memoria disponible de 512 MB. Ello implicó un enorme desafío. Para lograr el correcto funcionamiento, se redujeron en algunos casos los tamaños de los datasets. 
Como consecuencia inmediata, podemos inferir que los resultados de las consultas ya no serán los mismos. Independientemente de ello, demostramos que es posible el desarrollo solicitado.

Para el sistema de recomendación por similitud del coseno no se utilizó la columna de análisis de sentimiento. En vez de ello intenté utilizar las reviews de los usuarios para alimentar al sistema de procesamiento de texto. Me parecía mucho mas completa la información obtenida de esta forma. Consecuencia inmediata fue el tamaño gigantesco de la matriz generada. Se descartaron aleatoriamente filas del dataset original para lograr su funcionamiento. 

Finalmente se logró el deploy en Render, luego de innumerables pruebas en la reducción de los datasets.

La limitación del tamaño de subido de los archivos a GitHub se solucionó utilizando parquet como formato de compresión.

Posibles mejoras

El EDA adjunto en el archivo eda.ipynb es básico y puede mejorarse tanto en extensión como en presentación.

La matriz generada en el sistema de ML puede trabajarse para reducir su tamaño, utilizando alguna de las técnicas disponibles para tal fin.

El código se podría optimizar para reducir su extensión y performance.

Una versión paga de Render ayudaría a un mejor desempeño del deploy.


Archivos y carpetas:

En el directorio raíz se encuentran los archivos básicos para el funcionamiento de la API. En la carpeta Notebooks están los Jupyter Notebooks de las funciones y también se replica la carpeta Tablas (también presente en el directorio raíz) para que puedan ejecutarse las funciones utilizando la ruta relativa correcta de los datasets.










