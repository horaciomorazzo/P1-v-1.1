
from fastapi import FastAPI

import api_functions as af
import importlib
importlib.reload(af)

# Se instancia la aplicación
app = FastAPI()

# Llamada a las funciones desde FastAPI

@app.get("/playtimegenre/{genero}")
async def PlayTimeGenre(genero: str):
    
    return af.PlayTimeGenre(genero)

@app.get("/userforgenre/{genero}")
async def UserForGenre(genero: str):

    return af.UserForGenre(genero) 

@app.get("/usersrecommend/{anio}")
async def UsersRecommend(anio: int):

    return af.UsersRecommend(anio)
    
@app.get("/usersworstdeveloper/{anio}")
async def UsersWorstDeveloper(anio: int):

    return af.UsersWorstDeveloper(anio)

@app.get("/sentiment_analysis/{developer}")
async def sentiment_analysis(developer: str):

    return af.sentiment_analysis(developer)

@app.get("/recomendacion_juego/{product_id}")
async def sentiment_analysis(product_id: str):

    return af.recomendacion_juego(product_id)