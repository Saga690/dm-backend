from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import analyze_stock  # Import your existing function
import os

# Initialize FastAPI app
app = FastAPI()


origins=["https://stock-ai-frontend-git-main-divyanshs-projects-b99a3826.vercel.app"
         ,"https://stock-ai-frontend.vercel.app",
         "http://stock-ai-frontend.vercel.app",
        "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow frontend URL
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define a request model
class StockQuery(BaseModel):
    query: str

# Define a route for stock analysis
@app.post("/analyze_stock")
def analyze_stock_endpoint(stock_query: StockQuery):
    try:
        response = analyze_stock(stock_query.query)  # Call the function
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
def root():
    return {"message": "Stock Analysis API is running!"}


