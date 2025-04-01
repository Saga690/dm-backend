from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Any
from tools import StockDataRetriever,StockRAG,TickerExtractionTool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

import html

load_dotenv()  # Load variables from .env
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.1
    )

class StockAgentState(TypedDict):
    """State for the stock agent workflow"""
    query: str
    ticker: str
    stock_data: str
    response: str
    error: str


def extract_ticker_node(state: StockAgentState) -> StockAgentState:
    """Node for extracting ticker from the query"""
    try:
        ticker_tool = TickerExtractionTool(llm)
        ticker = ticker_tool.extract_ticker(state["query"])
        
        if ticker == "UNKNOWN":
            return {**state, "error": "Could not identify a valid stock ticker in your query."}
        
        return {**state, "ticker": ticker}
    except Exception as e:
        return {**state, "error": f"Error extracting ticker: {str(e)}"}

def retrieve_stock_data_node(state: StockAgentState) -> StockAgentState:
    """Node for retrieving stock data"""
    if state.get("error"):
        return state
    
    try:
        print("Retrieving stock data for ticker:", state["ticker"])
        ticker = state["ticker"]
        stock_retriever = StockDataRetriever()
        stock_data = stock_retriever.get_all_stock_data(ticker)
        
        return {**state, "stock_data": stock_data}
    except Exception as e:
        return {**state, "error": f"Error retrieving stock data: {str(e)}"}
    
def rag_node(state: StockAgentState) -> StockAgentState:
    """Node for RAG operations (store and retrieve)"""
    if state.get("error"):
        return state
    
    try:
        ticker = state["ticker"]
        query = state["query"]
        stock_data = state["stock_data"]
        
        # Initialize RAG system
        rag_system = StockRAG()
        
        # Store the stock data
        rag_system.store_stock_data(ticker, stock_data)
        
        # Retrieve and answer
        answer = rag_system.retrieve_and_answer(ticker, query)
        
        return {**state, "response": answer}
    except Exception as e:
        return {**state, "error": f"Error in RAG process: {str(e)}"}
    

def format_response_node(state):
    """Node for formatting the final response with HTML"""
    if state.get("error"):
        error_html = f"""
        <div class="error-message p-4 bg-red-500/20 border border-red-500 rounded-lg">
            <h3 class="text-lg font-bold text-red-500 mb-2">Error</h3>
            <p class="text-white">{state['error']}</p>
        </div>
        """
        return {**state, "response": error_html}
    
    ticker = state["ticker"]
    response = state["response"]
    
    # If response is not already HTML, wrap it in HTML
    if not response.strip().startswith("<"):
        escaped_response = html.escape(response).replace("\n", "<br>")  # Escape & format new lines

        formatted_response = f"""
        <div class="stock-analysis">
            <h2 class="text-xl font-bold text-emerald-500 mb-4">{ticker} Stock Analysis</h2>
            <div class="bg-gray-800/50 p-4 rounded-lg">
                {escaped_response}
            </div>
        </div>
        """
        return {**state, "response": formatted_response}

def decide_next_step(state: StockAgentState) -> str:
    """Decide which node to go to next based on the current state"""
    if state.get("error"):
        return "format_response"
    
    if not state.get("ticker"):
        return "extract_ticker"
    
    if not state.get("stock_data"):
        return "retrieve_stock_data"
    
    if not state.get("response"):
        return "rag"
    
    return "format_response"


def build_stock_analysis_graph():
    """Build the LangGraph workflow for stock analysis"""
    # Create a new graph
    workflow = StateGraph(StockAgentState)
    
    # Add the nodes
    workflow.add_node("extract_ticker", extract_ticker_node)
    workflow.add_node("retrieve_stock_data", retrieve_stock_data_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("format_response", format_response_node)
    
    # Add conditional edges based on the decide_next_step function
    workflow.add_conditional_edges(
        "extract_ticker",
        decide_next_step,
        {
            "retrieve_stock_data": "retrieve_stock_data",
            "format_response": "format_response"
        }
    )
    
    workflow.add_conditional_edges(
        "retrieve_stock_data",
        decide_next_step,
        {
            "rag": "rag",
            "format_response": "format_response"
        }
    )
    
    workflow.add_conditional_edges(
        "rag",
        decide_next_step,
        {
            "format_response": "format_response"
        }
    )
    
    # Add the final edge
    workflow.add_edge("format_response", END)
    
    # Set the entry point
    workflow.set_entry_point("extract_ticker")
    
    # Compile the graph
    return workflow.compile()

def analyze_stock(query: str):
    """Run the stock analysis agent with a query"""
    # Build the graph
    graph = build_stock_analysis_graph()
    
    # Create the initial state
    initial_state = StockAgentState(
        query=query,
        ticker="",
        stock_data="",
        response="",
        error=""
    )
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Return the final response
    return result["response"]

if __name__ == "__main__":
    # Ask the user for a query
    user_query = input("Ask any market-related question: ")
    
    # Run the stock analysis agent
    response = analyze_stock(user_query)
    
    # Print the response
    print("\nResponse:\n", response)

