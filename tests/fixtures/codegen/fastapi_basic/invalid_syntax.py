from fastapi import FastAPI

app = FastAPI(  # syntax error: unclosed paren
    @app.get("/")
def read_root():
    return {"Hello": "World"}