from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import analyze_stock  
import os
import asyncio

# Initialize FastAPI app
app = FastAPI()

# Define allowed origins
origins = [
    "https://stock-ai-frontend-git-main-divyanshs-projects-b99a3826.vercel.app",
    "https://stock-ai-frontend.vercel.app",
    "http://stock-ai-frontend.vercel.app",
    "http://localhost:5173",
    "https://dm-frontend-self.vercel.app",
    "https://dm-frontend-self.vercel.app/",
    "http://dm-frontend-self.vercel.app",
    "http://dm-frontend-self.vercel.app/"
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Define a request model
class StockQuery(BaseModel):
    query: str

# Define a route for stock analysis
@app.post("/analyze_stock")
async def analyze_stock_endpoint(stock_query: StockQuery, request: Request):
    try:
        # Set a timeout of 30 seconds for the analysis
        async def run_analysis():
            return analyze_stock(stock_query.query)
        
        try:
            response = await asyncio.wait_for(run_analysis(), timeout=300.0)
            return {"response": response}
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Request timed out. The analysis is taking longer than expected. Please try again."
            )
            
    except Exception as e:
        # Log the full error for debugging
        print(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

# Root endpoint
@app.get("/")
def root():
    return {"message": "Stock Analysis API is running!"}


