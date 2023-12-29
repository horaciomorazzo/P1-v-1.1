## FUNCIONES A UTILIZAR EN app.py

# Importaciones
import pandas as pd
import operator
import tools
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords') 
import sklearn



# Leo las tablas a utilizar

df_steam_games_output = pd.read_parquet('Tablas/steam_games_output_limpio.parquet')
df_user_items = pd.read_parquet('Tablas/user_items_limpio.parquet')
df_user_reviews = pd.read_parquet('Tablas/user_review_final.parquet')
df_user_reviews_sa = pd.read_parquet('Tablas/user_review_sa.parquet')
f4aux = pd.read_parquet('Tablas/f4aux.parquet')
f5aux = pd.read_parquet('Tablas/f5aux.parquet')
ml_genres = pd.read_parquet('Tablas/ml_genres.parquet')
ml_reviews = pd.read_parquet('Tablas/ml_reviews.parquet')
ml_title_item_id = pd.read_parquet('Tablas/ml_reviews.parquet')


# Funciones


def PlayTimeGenre(genero):

    """ 
        Devuelve el año con mas horas jugadas para el género ingresado.

        Parámetro de entrada: genero (str)
        Parámetro de salida: respuesta (str)

    """
    # Empiezo a trabajar con user_items para reducir su tamaño
    df_b = df_user_items
    # Elimino las columnas que no voy a utilizar
    df_b.drop(columns=['user_id', 'item_name'])
    # Sumo la cantidad de horas jugadas por item_id
    a = df_b.groupby('item_id')['playtime_forever'].sum()
    # Transformo la serie en dataframe
    df_playtime = a.to_frame()
    df_playtime = df_playtime.reset_index(drop=False)
    # Leo el siguiente dataframe: user_review_final    
    df_ur1 = df_user_reviews    
    # Elimino las columnas que no voy a utilizar
    df_ur2 =df_ur1.drop(columns=['user_id', 'helpful', 'recommend', 'review'])
    # Elimino las filas donde posted_year = 0
    df_ur3 = df_ur2.drop(df_ur2[df_ur2['posted_year'] == 0].index)
    df_ur3.reset_index(drop=True, inplace=True)
    df_ur3['item_id'] = df_ur3['item_id'].astype(int)
    # Realizo un join de df_ur3 y df_playtime utilizando la columna item_id
    df_join1 = df_ur3.merge(df_playtime, on="item_id", how="inner")
    # Cambio el tipo de dato de 'item_id' a entero
    df_join1['item_id'] = df_join1['item_id'].astype(int)
    # Leo el dataframe steam_games_output
    df_sgo = df_steam_games_output
    # Elimino del dataframe todas las filas donde en  la columna release_year aparece 'Dato no disponible'
    df_sgo1 = df_sgo[df_sgo["release_year"] != "Dato no disponible"]
    # Reinicio índice
    df_sgo1.reset_index(drop=True, inplace=True)
    # Conservo las columnas 'genres' e 'id'
    df_sgo1 = df_sgo1.drop(columns=['title', 'developer', 'release_year'])
    # Renombro la columna 'id' para no tener problemas en el join
    df_sgo1.rename(columns={'id': 'item_id'}, inplace=True)
    # Cambio el tipo de dato de item_id a entero
    df_sgo1['item_id'] = df_sgo1['item_id'].astype(int)
    # Ahora estoy en condiciones de hacer el join
    df_join2 = df_sgo1.merge(df_join1, on="item_id", how="inner")
    # Agrupo el dataframe por 'genres' y obtengo el índice del valor máximo de 'playtime_forever', es decir
    # del máximo de horas jugadas por género.
    idx = df_join2.groupby('genres')['playtime_forever'].idxmax()
    # Me quedo con una lista de los géneros 
    result = df_join2.loc[idx, ['genres', 'posted_year', 'playtime_forever']]
    # El siguiente código filtra en el dataframe 'result' el año correspondiente al género que yo ingreso. 
    df_filtrado = result.loc[result["genres"] == genero, "posted_year"]
    if df_filtrado.empty:
        return "Género inexistente. Los géneros válidos para ingresar son: Action, Adventure, Audio Production, Casual, Early Access, Education, Free to Play, Indie, Massively Multiplayer, RPG, Racing, Simulation, Software Training, Sports, Strategy, Utilities, Video Production, Web Publishing. RESPETAR MAYÚSCULAS Y ESPACIOS ENTRE LAS PALABRAS."
    else:
        a = df_filtrado.to_string(index=False)
        respuesta = f"{{Año de lanzamiento con más horas jugadas para Género {genero} : {a}}}"

        return respuesta


def UserForGenre(genero):
     
     """ 
        Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

        Parámetro de entrada: genero (str)
        Parámetro de salida: cadena (str)

    """
     
     # Leo el dataframe user_reviews_final
     df_ur1 = df_user_reviews
     # Elimino las columnas que no voy a utilizar
     df_ur2 = df_ur1.drop(columns=['helpful', 'recommend', 'review' ])
     # Cambio el nombre de la columna item_id por id
     df_ur2.rename(columns={'item_id': 'id'}, inplace=True) 
     # Elimino filas donde posted_year = 0
     condicion = df_ur2['posted_year'] == 0
     df_ur2.drop(df_ur2[condicion].index, inplace=True)
     # Cambio el tipo de dato de la columna 'id' a entero
     df_ur2['id'] = df_ur2['id'].astype(int)
     # Leo steam_games_output
     df_sgo = df_steam_games_output
     # Me quedo con 'genres' e 'id'
     df_sgo1 =df_sgo.drop(columns=['title', 'developer', 'release_year' ])    
     df_sgo1['id'] = df_sgo1['id'].astype(int)
     # Ahora hago la unión con df_ur2, así me quedan los usuarios relacionados con el año
     df_join2 = pd.merge(df_sgo1, df_ur2, on='id')
     # Leo el dataframe user_items
     df_ui = df_user_items
     # No necesito item_name. Elimino la columna
     df_ui1 =df_ui.drop(columns=['item_name' ])
     # cambio el nombre de la columna item_id por id
     df_ui1.rename(columns={'item_id': 'id'}, inplace=True)
     df_ui1['id'] = df_ui1['id'].astype(int)
     # Hago un join de df_ui1 con df_sgo1
     df_join1 = df_ui1.merge(df_sgo1, on="id", how="inner")
     # Sumo el tiempo jugado para cada género y jugador
     # Ignoro la unidad de tiempo de los registros de playtime_forever. Dado que la respuesta que se pide es comparativa (medición relativa, no absoluta), el resultado no se verá afectado por la unidad elegida, asumiendo que todos los registros de la columna 'playtime_forever' comparten la misma unidad de medida. 
     df_grouped = df_join1.groupby(["user_id", "genres"])["playtime_forever"].sum()
     df_grouped = df_grouped.to_frame().rename(columns={'playtime_forever': 'suma_play'})
     df_new = df_grouped.reset_index()
     # Hallo los índices que corresponden al máximo de tiempo jugado de cada género
     idx = df_new.groupby("genres")["suma_play"].idxmax()
     # Me quedo con una lista de los géneros y usuarios correspondientes al género que tiene más tiempo jugado
     result = df_new.loc[idx, ['user_id', 'genres', 'suma_play']]
     result.reset_index(drop=True, inplace=True)
     # El siguiente código filtra en el dataframe 'result' el año correspondiente al género que yo ingreso. 
     df_filtrado = result.loc[result["genres"] == genero, "user_id"]
     # Hallo el usuario de interés y luego transformo el dataframe en string
     usuario = df_filtrado.to_string(index=False)
     # Ahora tengo que hallar el acumulado de horas jugadas por año para los usuarios del dataframe df_filtrado únicamente.
     # Me quedo con la lista de usuarios de interés.
     # En df_join2 tengo que filtrar los usuarios de interés. Los tomo de result. Previamente elimino a los repetidos.
     result_unicos = result.drop_duplicates(subset="user_id")
     # Ahora me quedo solo con los user_id y reinicio el índice
     result_unicos = result_unicos.drop(["genres", "suma_play"], axis=1)
     result_unicos.reset_index(drop=True, inplace=True)
     # Filro df_join2 para quedarme solo con los usuarios que me interesan
     df_prueba = df_join2[df_join2["user_id"].isin(result_unicos["user_id"])]
     # Hago un join con result
     df_join3 = df_prueba.merge(result, on="genres", how="inner")
     # Elimino las columnas user_id_x y suma_play (no me sirven)
     df_join3.drop(columns=['user_id_x', 'suma_play'])
     # De df_ui1 me quedo con las columnas que voy a utilizar
     df_ui2 = df_ui1.drop(columns=['user_id'])
     # Termino de armar el dataframe definitivo
     df_final = df_join3.merge(df_ui2, on='id', how='inner')
     df_final1 = df_final.drop(columns=['user_id_x', 'suma_play'])
     # Filtramos por usuario(usuario) y género(genero).
     df_filtrado = df_final1.query("genres == @genero & user_id_y == @usuario")
     # Sumamos los minutos de cada año (supongo que el tiempo está expresado en minutos)
     df_agrupado = df_filtrado.groupby("posted_year")["playtime_forever"].sum()
     # Lo transformamos en dataframe
     df_nuevo = df_agrupado.reset_index()
     # Dividimos por 60 para que 'playtime_forever' quede efectivamente expresado en horas
     df_nuevo['playtime_forever'] = (df_nuevo['playtime_forever']/60).astype(int)
     # Reemplazo nombre de columnas por los pedidos en el proyecto
     df = df_nuevo.rename(columns={"posted_year": "Año", "playtime_forever": "Horas"})
     # Para formatear la respuesta necesito transformar enteros en string
     df = df.astype({"Año": str, "Horas": str})
     # Convierto el dataframe en una lista de diccionarios
     lista = df.to_dict(orient='records')
     # Convierto la lista de diccionarios en una cadena de caracteres, para poder eliminar las comillas simples.
     cadena = str(lista)
     # Reemplazo las comillas simples por vacíos
     cadena = cadena.replace("'", "")
     cadena = f'{{"Usuario con más horas jugadas para Género {genero}" : {usuario}, "Horas jugadas":{cadena}}}'

     return cadena

      
def UsersRecommend(anio):

    """ 
        Devuelve el top 3 de juegos más recomendados por usuarios para el año dado. 
        (reviews.recommend = True y comentarios positivos/neutrales)

        Parámetro de entrada: anio (int)
        Parámetro de salida: resultado (str)

    """

    # Leemos el dataframe user_reviews_sa (No se eliminaron filas con recomendaciones nulas)
    df_ur1 = df_user_reviews_sa
    # Elimino la columna 'helpful'
    df2 = df_ur1.drop("helpful", axis=1) 
    # Cambiamos el True y False de la columna 'recommend' por valores enteros, 1 y 0 respectivamente.
    df2["recommend"] = df2["recommend"].astype(int)  
    # Ahora trabajaremos con la columna 'review' y la función a_s (análisis de sentimientos)
    df2['sentiment'] = df2['review'].apply(tools.a_s) 
    # Elimino las columnas user_id y review pues no las necesito para la función solicitada
    df3 = df2.drop(["user_id", "review"], axis=1)
    df3['item_id'] = df3['item_id'].astype(int)
    df4 = df3.rename(columns={'item_id':'id'})
    # Leo el dataframe steam_games_output
    df_sgo = pd.read_parquet('Tablas/steam_games_output_limpio.parquet')
    # Elimino del dataframe todas las filas donde en  la columna release_year aparece 'Dato no disponible'
    df_sgo1 = df_sgo[df_sgo["release_year"] != "Dato no disponible"]
    # Me quedo con las columnas 'genres' e 'id'
    df_sgo1 = df_sgo1.drop(columns=['genres', 'developer', 'release_year'])
    # Elimino duplicados
    df_sgo2 = df_sgo1.drop_duplicates()
    # Reinicio índice
    df_sgo2.reset_index(drop=True, inplace=True)
    # Cambio el tipo de dato de 'id' a entero
    df_sgo2['id'] = df_sgo2['id'].astype(int)
    # Hago un join con df3
    df_join = df_sgo2.merge(df4, on="id", how="inner")
    # Filtro el dataframe por el año y el sentimiento
    df_filtrado = df_join[(df_join["posted_year"] == anio) & (df3["sentiment"] != 0)]
    # Agrupo el dataframe por el item_id y sumo las recomendaciones
    df_agrupado = df_filtrado.groupby("title")["recommend"].sum()
    # Ordeno el dataframe de forma descendente y obtengo los tres primeros valores
    df_ordenado = df_agrupado.sort_values(ascending=False).head(3)
    # Formateo el resultado para presentarlo como es solicitado
    lista = df_ordenado.reset_index().to_dict("records")
    resultado = [{"Puesto {0}".format(i+1): x["title"]} for i, x in enumerate(lista)]
    
    return resultado


def UsersWorstDeveloper(anio):

     """ 
        Devuelve el top 3 de desarrolladoras con juegos menos recomendados por usuarios para el año dado. 
        (reviews.recommend = False y comentarios negativos)
        
        Parámetro de entrada: anio (int)
        Parámetro de salida: df_list (str)

     """
     
     # Necesito relacionar item_id con el developer. Leo el dataframe steam_output
     df1 = df_steam_games_output 
     # Me quedo con las columnas 'id' y 'developer'
     df2 = df1.drop(["genres", "release_year", "title"], axis=1)
     # Elimino duplicados
     df2.drop_duplicates(subset='id', inplace=True)
     df2.reset_index(drop=True, inplace=True)
     # Cambio el tipo de dato a entero
     df2['id'] = df2['id'].astype(int)
     # Leo el dataframe auxiliar f4aux y renombro la columna 'id'
     df3 = f4aux
     df2.rename(columns={"id":"item_id"}, inplace=True)
     # Uno df2 con df3
     df4 = pd.merge (df2, df3, on="item_id", how="inner")
     #Filtro el dataframe por recommend = 0 y sentiment = 0
     df_filtered = df4[(df4["recommend"] == 0) & (df4["sentiment"] == 0)]
     #Elijo el año que me interesa
     df_year = df_filtered[df_filtered["posted_year"] == anio]
     #Calculo el top 3 de developer 
     df_top3 = df_year.groupby("developer")["developer"].count().nlargest(3)
     #Muestro resultados con la visualización solicitada
     df_dict = df_top3.to_dict()
     df_list = [{f"Puesto {i+1}" : v} for i, v in enumerate(df_dict)]

     return df_list
     
    
def sentiment_analysis(developer):

     """ 
        Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor.
        
        Parámetro de entrada: developer (str)
        Parámetro de salida: respuesta (str)

     """

     # Leo el dataframe auxiliar f5aux
     df = f5aux
     # Creo columnas con los 3 valores diferentes de sentiment
     df_pivot = df.pivot_table(index="developer", columns="sentiment", values="item_id", aggfunc="count")
     # Reemplazo nulos con ceros
     a = df_pivot.fillna(0)
     # Transformo todos los datos a enteros
     a = a.astype(int)
     # Busco los valores de sentiment para el developer ingresado
     a0 = a.loc[developer, 0]
     a1 = a.loc[developer, 1]
     a2 = a.loc[developer, 2]
     # Devuelvo la respuesta en el formato requerido
     respuesta = f"{{\'{developer}\':[Negative={a0}, Neutral={a1}, Positive={a2}]}}"

     return respuesta



def recomendacion_juego(product_id):

    try:
        """ 
            
            Ingresando el id de producto, se devuelve una lista con 5 juegos recomendados similares al ingresado.
            
            Parámetro de entrada: product_id (str)
            Parámetro de salida: respuesta (list)

        """
        
        # Lectura de datasets
        df1 = pd.read_parquet('Tablas/ml_genres.parquet')
        df2 = pd.read_parquet('Tablas/ml_reviews.parquet')
        df4 = pd.read_parquet('Tablas/ml_title_item_id.parquet')
        # Renombro y cambio tipo de columna para que funcionen los join's
        df1.rename(columns={"id": "item_id"}, inplace=True)
        df1['item_id'] = df1['item_id'].astype(str)
        # Me quedo con las columnas que voy a necesitar únicamente
        df3 = df2.drop(columns=['posted', 'last_edited', 'funny', 'helpful'])
        # Reemplazo nulos por vacíos. Unifico tipo de datos para join
        df3.fillna('')
        df3['item_id'] = df3['item_id'].astype(str)
        # Armo el dataset que voy a necesitar
        df_join1 = df4.merge(df1, on="item_id", how="inner")
        df_join = df3.merge(df_join1, on="item_id", how="inner")
        # Reemplazo nulos por vacíos
        df5 = df_join.fillna('')
        # Elimino duplicados
        df5.drop_duplicates()
        # Cambio tipo bool a string, para luego procesar como texto
        df5['recommend'] = df5['recommend'].astype(str)
        # Elimino aleatoriamente filas de df5. Me quedo con el 10%, para evitar problemas con el uso de memoria.
        df5 = df5.sample(frac = 0.1)
        # Reseteo índice
        df5.reset_index(drop=True, inplace=True)
        # Creamos un nuevo dataframe con una única columna donde cada fila concatena los contenidos de todas las columnas que nos interesan.
        todo = []
        for i in range(0, df5.shape[0]):
            todo.append(df5['genres'][i]+' '+df5['item_id'][i]+' '+df5['title'][i]+' '+df5['review'][i])
        df5['todo'] = todo
        # Creamos nuestro propio índice
        df5.insert(1, "id", list(range(1, 12000)), True) 
        # Agregamos el índice como primer columna del dataset
        df_new = df5[['id','todo']]
        # Eliminamos conectores (a, an, are, etc.)
        from nltk.corpus import stopwords
        stop = stopwords.words('english')
        def text_preprocessing(column):
            # Convertimos a minúsculas
            column = column.str.lower()
            # Convertimos puntuaciones y símbolos extraños en vacíos
            column = column.str.replace('http\S+|www.\S+|@|%|:|,|', '', case=False)
            # Dividimos oraciones en palabras para aplicar las funciones previas
            word_tokens = column.str.split()
            keywords = word_tokens.apply(lambda x: [item for item in x if item not in stop])
            # Rearmamos las oraciones y les asignamos una nueva columna
            for i in range(len(keywords)):
                keywords[i] = " ".join(keywords[i])
                column = keywords

            return column
        # Creamos una nueva columna con el texto ya procesado
        df_new['cleaned_infos'] = text_preprocessing(df_new['todo'])
        # Aplicamos similaridad del coseno
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        # convierto una colección de documentos de texto en una matriz de conteo de tokens usando la clase CountVectorizer de la librería sklearn. Esta clase me permite extraer las características de los textos, como las palabras, los n-gramas, el vocabulario, etc. y crear una representación numérica de los mismos. El método fit_transform() combina los pasos de ajustar el modelo a los datos y transformar los datos en la matriz de conteo de tokens. El resultado es una matriz dispersa que contiene los valores de frecuencia de cada token en cada documento.
        CV = CountVectorizer()
        converted_matrix = CV.fit_transform(df_new['cleaned_infos'])
        # Utilizamos la función cosine_similarity que nos devuelve una matriz cuadrada que contiene los valores de similitud de coseno entre cada par de vectores de la matriz de entrada. 
        cosine_similarity = cosine_similarity(converted_matrix)
        # Ingresamos un item_id para hallar recomendaciones
        input_item = product_id
        item_id = df5[df5['item_id'] == input_item]['id'].values[0]
        # Obtengo una lista de pares (índice, valor) que representan la similitud de coseno entre el elemento con el índice item_id y cada uno de los demás elementos de la matriz cosine_similarity. La función enumerate() toma un iterable y devuelve un objeto que genera los pares (índice, valor) para cada elemento del iterable. La función list() convierte el objeto en una lista
        score = list(enumerate(cosine_similarity[item_id]))
        # Ordeno la lista score de forma descendente según el segundo elemento de cada par (índice, valor). La función sorted() toma un iterable y devuelve una lista ordenada. El argumento key permite especificar una función que se aplica a cada elemento del iterable antes de compararlo. El argumento reverse permite indicar si se quiere ordenar de forma ascendente (False) o descendente (True). En este caso, se usa una función lambda como valor de key, que toma como entrada x y devuelve x1, es decir, el segundo elemento de x
        sorted_score = sorted(score, key=lambda x:x[1], reverse= True)
        # Elimino el primer elemento de la lista pues obviamente el ítem más parecido al ítem ingresado es el mismo ítem.
        sorted_score = sorted_score[1:]
        # Ítems más parecidos al ingresado, primeros 5, orden descendente.
        i = 0
        lista = []
        for item in sorted_score:
            items = df5[df5['id'] == item[0]]['item_id'].values[0]
            #print(i+1,items)
            lista.append((i+1, items))
            i = i+1
            if i > 4:
                break
        return(lista)
    except:
        return "Identificador de producto desconocido"

