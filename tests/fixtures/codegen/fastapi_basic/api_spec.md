# FastAPI — basic app

Create an application with the `FastAPI` class:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```

A path operation decorator (`@app.get`) registers a handler. Handlers are plain
functions; the return value is serialized as JSON.

Serve it locally with uvicorn:

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```