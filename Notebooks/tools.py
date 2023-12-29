import pandas as pd
from textblob import TextBlob
import re

def a_s(texto):
    '''
    Realiza un análisis de sentimiento en un texto dado clasificándolo numéricamente en 3 categorías: 

     0: Sentimiento negativo
     1: Sentimiento neutral
     2: Sentimiento positivo

    Utilizamos la librería TextBlob para analizar el sentimiento.

    Parámetro de entrada:
    
    texto (str): El texto que se va a analizar.

    Parámetro devuelto:

    Un valor numérico entero (int)
    '''
    if texto is None:
        return 1
    texto_procesado = TextBlob(texto)
    polarity = texto_procesado.sentiment.polarity
    if polarity < -0.2:
        return 0  
    elif polarity > 0.2: 
        return 2 
    else:
        return 1 