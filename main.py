from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import analyze_stock  
import os
import asyncio
from fastapi.responses import JSONResponse

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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
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
            response = await asyncio.wait_for(run_analysis(), timeout=3000.0)
            return JSONResponse(
                content={"response": response},
                headers={
                    "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out. The analysis is taking longer than expected. Please try again."},
                headers={
                    "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
            
    except Exception as e:
        # Log the full error for debugging
        print(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred while processing your request: {str(e)}"},
            headers={
                "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true"
            }
        )

# Root endpoint
@app.get("/")
def root():
    return {"message": "Stock Analysis API is running!"}

# Add OPTIONS endpoint explicitly
@app.options("/analyze_stock")
async def options_analyze_stock(request: Request):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true"
        }
    )


