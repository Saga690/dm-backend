import json
import yfinance as yf
from scraping import fetch_stock_news_links, extract_full_stock_news
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.tools import Tool
import html
import time
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()  
groq_api_key = os.getenv("GROQ_API_KEY")

# Rate limiting configuration
class RateLimiter:
    def __init__(self, max_requests=30, time_window=60):  # 30 requests per minute
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def wait_if_needed(self):
        now = datetime.now()
        # Remove requests older than the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(seconds=self.time_window)]
        
        if len(self.requests) >= self.max_requests:
            # Wait until the oldest request is outside the time window
            sleep_time = (self.requests[0] + timedelta(seconds=self.time_window) - now).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.requests.append(now)

# Initialize rate limiter
rate_limiter = RateLimiter()

def get_yfinance_data(ticker, retries=3, delay=2):
    """Get Yahoo Finance data with retries and rate limiting"""
    for attempt in range(retries):
        try:
            rate_limiter.wait_if_needed()
            stock = yf.Ticker(ticker)
            return stock
        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < retries - 1:
                print(f"Rate limit hit, waiting {delay} seconds before retry {attempt + 1}/{retries}")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e

# Define the TickerExtractionTool
class TickerExtractionTool:
    def __init__(self, llm):
        self.llm = llm
        self.extract_ticker_tool = Tool(
            name="Extract Ticker",
            func=self.extract_ticker,
            description="Extracts the stock ticker symbol from a user query."
        )

    def extract_ticker(self, query):
        """Extracts the correct stock ticker from a given user query."""
        prompt = (
            "Your task is to extract the correct stock ticker symbol from the given user query. "
            "Follow these rules strictly: "
            "1. Return only the official stock ticker symbol, nothing else. "
            "2. If multiple companies are mentioned, return only the most relevant one. "
            "3. Ensure the ticker is valid and exists on major stock exchanges (NYSE, NASDAQ, etc.). "
            "4. If no valid company is found, return 'UNKNOWN'.\n\n"
            f"User Query: {query}\n\n"
            "Response (only the ticker, e.g., 'AAPL', 'GOOGL', 'TSLA', or 'UNKNOWN'):"
        )

        response = self.llm.invoke(prompt)
        ticker = response.content.strip().upper()
        return ticker


class StockDataRetriever:
    def __init__(self, output_dir="./stock_data"):
        """Initialize the stock data retriever"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def get_company_info(self, ticker):
        """Get basic company information"""
        try:
            stock = get_yfinance_data(ticker)
            info = stock.info
            
            # Extract only the relevant fields
            company_info = {
                'name': info.get('shortName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A'),
                'country': info.get('country', 'N/A'),
                'city': info.get('city', 'N/A')
            }
            
            return company_info
        except Exception as e:
            print(f"Error fetching company info for {ticker}: {e}")
            return {}
    
    def get_key_metrics(self, ticker):
        """Get key financial metrics for a ticker"""
        try:
            stock = get_yfinance_data(ticker)
            info = stock.info
            
            # Extract only key financial metrics
            metrics = {
                'currentPrice': info.get('currentPrice', 'N/A'),
                'marketCap': info.get('marketCap', 'N/A'),
                'peRatio': info.get('trailingPE', 'N/A'),
                'forwardPE': info.get('forwardPE', 'N/A'),
                'eps': info.get('trailingEps', 'N/A'),
                'dividendYield': info.get('dividendYield', 'N/A'),
                'bookValue': info.get('bookValue', 'N/A'),
                'priceToBook': info.get('priceToBook', 'N/A'),
                'profitMargins': info.get('profitMargins', 'N/A'),
                'returnOnEquity': info.get('returnOnEquity', 'N/A'),
                'returnOnAssets': info.get('returnOnAssets', 'N/A'),
                'debtToEquity': info.get('debtToEquity', 'N/A'),
                'currentRatio': info.get('currentRatio', 'N/A'),
                'quickRatio': info.get('quickRatio', 'N/A'),
                'beta': info.get('beta', 'N/A')
            }
            
            return metrics
        except Exception as e:
            print(f"Error fetching key metrics for {ticker}: {e}")
            return {}
    
    def get_financial_statements(self, ticker):
        """Get financial statements for a ticker"""
        try:
            stock = get_yfinance_data(ticker)
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Convert to dictionary with just the key metrics
            financials = {}
            
            # Income Statement - select key metrics
            if not income_stmt.empty:
                income_dict = {}
                key_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
                for metric in key_metrics:
                    if metric in income_stmt.index:
                        income_dict[metric] = {str(k): v for k, v in income_stmt.loc[metric].to_dict().items()}
                financials['income_statement'] = income_dict
            
            # Balance Sheet - select key metrics
            if not balance_sheet.empty:
                balance_dict = {}
                key_metrics = ['Total Assets', 'Total Current Assets', 'Total Liabilities', 
                              'Total Current Liabilities', 'Total Stockholder Equity']
                for metric in key_metrics:
                    if metric in balance_sheet.index:
                        balance_dict[metric] = {str(k): v for k, v in balance_sheet.loc[metric].to_dict().items()}

                financials['balance_sheet'] = balance_dict
            
            # Cash Flow - select key metrics
            if not cash_flow.empty:
                cash_dict = {}
                key_metrics = ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow']
                for metric in key_metrics:
                    if metric in cash_flow.index:
                        cash_dict[metric] = {str(k): v for k, v in cash_flow.loc[metric].to_dict().items()}
                financials['cash_flow'] = cash_dict
            
            return financials
        except Exception as e:
            print(f"Error fetching financials for {ticker}: {e}")
            return {}
    
    def get_historical_prices(self, ticker, period="2mo", interval="1d"):
        """Get historical price data for a ticker"""
        try:
            stock = get_yfinance_data(ticker)
            hist = stock.history(period=period, interval=interval)
            
            # Convert to dictionary
            if not hist.empty:
                hist_dict = {
                    'dates': [d.strftime('%Y-%m-%d') for d in hist.index],
                    'open': hist['Open'].tolist(),
                    'high': hist['High'].tolist(),
                    'low': hist['Low'].tolist(),
                    'close': hist['Close'].tolist(),
                    'volume': hist['Volume'].tolist()
                }
                return hist_dict
            return {}
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return {}
    
    def get_analyst_recommendations(self, ticker):
        """Get analyst recommendations for a ticker"""
        try:
            stock = get_yfinance_data(ticker)
            recommendations = stock.recommendations
            
            if recommendations is not None and not recommendations.empty:
                # Convert to dict with the last 10 recommendations
                recent_recommendations = recommendations.tail(10)
                rec_dict = {
                    'firm': recent_recommendations['Firm'].tolist(),
                    'to_grade': recent_recommendations['To Grade'].tolist(),
                    'from_grade': recent_recommendations['From Grade'].tolist(),
                    'action': recent_recommendations['Action'].tolist()
                }
                return rec_dict
            return {}
        except Exception as e:
            print(f"Error fetching recommendations for {ticker}: {e}")
            return {}
    
    def get_company_news(self, ticker):
        """Get recent news for a ticker"""
        try:
            news_links = fetch_stock_news_links(f"{ticker} stock news")
            full_news = extract_full_stock_news(news_links["articles"])
            return full_news["articles"]
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []
    
    def get_all_stock_data(self, ticker):
        """Get all data for a ticker and save to files"""
        print(f"Fetching all data for {ticker}...")
        
        # Create directory for this ticker
        ticker_dir = os.path.join(self.output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Fetch data with delays between calls
        company_info = self.get_company_info(ticker)
        time.sleep(1)  # Add delay between calls
        
        key_metrics = self.get_key_metrics(ticker)
        time.sleep(1)  # Add delay between calls
        
        financial_statements = self.get_financial_statements(ticker)
        time.sleep(1)  # Add delay between calls
        
        historical_prices = self.get_historical_prices(ticker)
        time.sleep(1)  # Add delay between calls
        
        analyst_recommendations = self.get_analyst_recommendations(ticker)
        time.sleep(1)  # Add delay between calls
        
        company_news = self.get_company_news(ticker)
        
        # Combine all data into one object
        all_data = {
            'company_info': company_info,
            'key_metrics': key_metrics,
            'financial_statements': financial_statements,
            'historical_prices': historical_prices,
            'analyst_recommendations': analyst_recommendations,
            'company_news': company_news
        }
        
        return json.dumps(all_data, indent=2)
    


class StockRAG:
    def __init__(self, db_path="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.chroma = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def store_stock_data(self, ticker, raw_text):
        chunks = self.text_splitter.split_text(raw_text)
        documents = [Document(page_content=chunk, metadata={"ticker": ticker}) for chunk in chunks]
        self.chroma.add_documents(documents)
        
    
    def retrieve_and_answer(self, ticker, query, top_k=5):
        retriever = self.chroma.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        retrieved_docs = [doc.page_content for doc in docs]
        
        if not retrieved_docs:
            return "I don't have enough data."
        
        context = "\n\n".join(retrieved_docs)
        prompt = f"""
        You are a highly skilled financial analyst with deep expertise in stock market analysis.
        Your task is to provide an accurate, well-structured, and visually appealing response based on the given financial data.

        ### **Stock Analysis for {ticker}**
        #### **Context:**
        {context}

        #### **Instructions:**
        1. **Use only the provided context** to formulate your response. Do not assume or generate information outside the given data.
        2. **Focus on key financial insights**, trends, and relevant stock metrics.
        3. **If data is insufficient**, clearly state the limitation rather than making assumptions.
        4. **Structure your response as HTML** with proper formatting for web display.
        5. **Include recommendations** if the user asks about stock market analysis or investment advice.
        6. **Explain complex financial terms** in a simple and understandable manner when necessary.
        7. **Use visual elements** to improve readability:
           - Use section headings with proper HTML heading tags (h2, h3, h4)
           - Format important metrics in bold
           - Use HTML tables for structured financial data
           - Create HTML lists with bullet points for key points
           - Include CSS classes that work with Tailwind CSS (text-emerald-500, bg-gray-800, etc.)

        #### **User's Question:**
        {query}

        #### **Response Format:**
        Your response MUST be valid HTML that can be directly rendered in a React component using dangerouslySetInnerHTML.
        Include proper HTML tags and structure your content with:
        - <div>, <h2>, <h3>, <p>, <table>, <ul>, <li> elements
        - Tailwind CSS classes for styling
        - Consistent spacing and organization
        
        DO NOT include any Markdown formatting or backticks - ONLY pure HTML that can be rendered directly.
        """
        llm = ChatGroq(
        model_name="llama3-70b-8192",  # You can change to another model like "mixtral-8x7b-32768"
        temperature=0.1,
        )
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def _process_html_response(self, response, ticker):
        """Process and clean the HTML response to ensure it's valid and well-structured"""
        # Check if response already contains HTML
        if "<div" not in response and "<h" not in response:
            escaped_response = html.escape(response).replace("\n", "<br>")  # Escape & format new lines
            # Convert non-HTML response to basic HTML
            return f"""
            <div class="stock-analysis">
                <h2 class="text-xl font-bold text-emerald-500 mb-4">{ticker} Analysis</h2>
                <div class="bg-gray-800/50 p-4 rounded-lg">
                    {escaped_response}
                </div>
            </div>
            """
        
        # Ensure the response has a wrapper div
        if not response.strip().startswith("<div"):
            response = f'<div class="stock-analysis">{response}</div>'
        
        return response