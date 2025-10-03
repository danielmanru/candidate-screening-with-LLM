from fastapi import FastAPI
from routes.routes import router

app = FastAPI(title="CV Evaluation API")
app.include_router(router)