from pymongo import AsyncMongoClient
import os
from dotenv import load_dotenv
load_dotenv()

client = AsyncMongoClient(os.getenv("MONGODB_URI"))
db = client["candidate-screening"]
