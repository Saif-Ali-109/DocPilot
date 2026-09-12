from fastapi import FastAPI

app = FastAPI()
app.add_middleware(ThrottleMiddleware, rate="10/s")