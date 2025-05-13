from fastapi import FastAPI
from src.api import query
from contextlib import asynccontextmanager
from src.retriever.retriever import load_data
from fastapi.responses import RedirectResponse
import logging


#uvicorn --host 0.0.0.0 src.main:app --reload
# TODO: Pre-load the dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename='app.log', filemode='a')
logger = logging.getLogger(__name__)



# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the data
    logger.info("Initializing application...")
    try:
        load_data()
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # We don't re-raise the exception to allow the app to start even if data loading fails
    
    # yield  # This is where the app runs
    
    # Shutdown: Clean up resources if needed
    logger.info("Shutting down application...")

app = FastAPI(
    title="ML API",
    description="API for ML Model Inference - Retrieves similar text excerpts based on semantic similarity",
    version="1.0.0",
)

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

app.include_router(query.router)

