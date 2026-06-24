from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.utils.loader import load_data
from app.routes import health, query, anomaly

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load dataset into memory on startup
    try:
        app.state.df = load_data()
        print(f"Data successfully loaded. Total rows: {len(app.state.df)}")
    except Exception as e:
        print(f"Critical Error: Failed to load dataset on startup: {e}")
    yield
    # Clean up on shutdown
    pass

app = FastAPI(
    title="Support Ticket Analytics System API",
    description="API backend for querying support ticket data, generating statistics, and detecting anomalies.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS so Streamlit frontend can communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["System Health"])
app.include_router(query.router, tags=["NLP Query Engine"])
app.include_router(anomaly.router, tags=["Anomaly Detection"])
