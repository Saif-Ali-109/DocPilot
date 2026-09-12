from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float

    @field_validator("name")
    def name_not_empty(cls, value):
        return value