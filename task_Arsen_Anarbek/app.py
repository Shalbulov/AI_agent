"""
FastAPI приложение для взаимодействия с AI агентом.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import os

from agent import process_query

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="AI Agent API",
    description="API для взаимодействия с AI агентом с MCP интеграцией",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production лучше указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем статические файлы
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class QueryRequest(BaseModel):
    """Модель запроса к агенту."""
    query: str


class QueryResponse(BaseModel):
    """Модель ответа от агента."""
    response: str
    query: str


@app.get("/")
async def root():
    """Корневой endpoint - возвращает веб-интерфейс."""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "AI Agent API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/v1/agent/query",
            "web_ui": "/static/index.html"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья приложения."""
    return {"status": "healthy"}


@app.post("/api/v1/agent/query", response_model=QueryResponse)
async def agent_query(request: QueryRequest):
    """
    Обрабатывает запрос пользователя через AI агента.
    
    Args:
        request: Запрос с текстом вопроса
        
    Returns:
        Ответ агента
        
    Raises:
        HTTPException: Если произошла ошибка при обработке запроса
    """
    try:
        logger.info(f"Получен запрос: {request.query}")
        
        # Обрабатываем запрос через агента
        response = process_query(request.query)
        
        logger.info(f"Ответ сгенерирован: {response[:100]}...")
        
        return QueryResponse(
            query=request.query,
            response=response
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
