"""
Тесты для AI агента и MCP сервера.
"""
import json
import os
import pytest
from pathlib import Path
import sys

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import (
    list_products,
    get_product,
    add_product,
    get_statistics,
    load_products,
    save_products,
    DATA_FILE
)
from agent import process_query, calculate_discount, format_currency
from app import app
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def setup_test_data():
    """Настройка тестовых данных перед каждым тестом."""
    # Сохраняем оригинальный файл, если существует
    original_exists = DATA_FILE.exists()
    if original_exists:
        with open(DATA_FILE, "r") as f:
            original_data = json.load(f)
    else:
        original_data = None
    
    # Создаем тестовые данные
    test_products = [
        {
            "id": 1,
            "name": "Ноутбук",
            "price": 50000,
            "category": "Электроника",
            "in_stock": True
        },
        {
            "id": 2,
            "name": "Смартфон",
            "price": 30000,
            "category": "Электроника",
            "in_stock": True
        }
    ]
    save_products(test_products)
    
    yield
    
    # Восстанавливаем оригинальные данные
    if original_data is not None:
        save_products(original_data)
    elif DATA_FILE.exists():
        DATA_FILE.unlink()


class TestMCPServer:
    """Тесты для MCP сервера."""
    
    def test_list_products_all(self):
        """Тест получения всех продуктов."""
        products = list_products()
        assert len(products) == 2
        assert products[0]["name"] == "Ноутбук"
        assert products[1]["name"] == "Смартфон"
    
    def test_list_products_by_category(self):
        """Тест получения продуктов по категории."""
        products = list_products(category="Электроника")
        assert len(products) == 2
        assert all(p["category"] == "Электроника" for p in products)
        
        products = list_products(category="Мебель")
        assert len(products) == 0
    
    def test_get_product_by_id(self):
        """Тест получения продукта по ID."""
        product = get_product(1)
        assert product["id"] == 1
        assert product["name"] == "Ноутбук"
        assert product["price"] == 50000
    
    def test_get_product_not_found(self):
        """Тест получения несуществующего продукта."""
        with pytest.raises(ValueError, match="не найден"):
            get_product(999)
    
    def test_add_product(self):
        """Тест добавления нового продукта."""
        new_product = add_product(
            name="Мышка",
            price=1500,
            category="Электроника",
            in_stock=True
        )
        
        assert new_product["id"] == 3
        assert new_product["name"] == "Мышка"
        assert new_product["price"] == 1500
        
        # Проверяем, что продукт сохранен
        products = load_products()
        assert len(products) == 3
        assert any(p["name"] == "Мышка" for p in products)
    
    def test_get_statistics(self):
        """Тест получения статистики."""
        stats = get_statistics()
        
        assert stats["count"] == 2
        assert stats["average_price"] == 40000.0  # (50000 + 30000) / 2


class TestAgentTools:
    """Тесты для кастомных инструментов агента."""
    
    def test_calculate_discount(self):
        """Тест вычисления скидки."""
        result = calculate_discount.invoke({"price": 1000, "discount_percent": 15})
        
        assert result["original_price"] == 1000
        assert result["discount_percent"] == 15
        assert result["discount_amount"] == 150
        assert result["final_price"] == 850
    
    def test_format_currency(self):
        """Тест форматирования валюты."""
        result = format_currency.invoke({"amount": 1234.56, "currency": "RUB"})
        assert "₽" in result
        assert "1234.56" in result


class TestAgentIntegration:
    """Тесты интеграции агента."""
    
    def test_agent_list_products(self):
        """Тест запроса списка продуктов через агента."""
        response = process_query("Покажи все продукты")
        assert "Ноутбук" in response or "Смартфон" in response
    
    def test_agent_get_statistics(self):
        """Тест запроса статистики через агента."""
        response = process_query("Какая средняя цена продуктов?")
        assert "40000" in response or "средняя" in response.lower()


class TestFastAPI:
    """Тесты для FastAPI приложения."""
    
    @pytest.fixture
    def client(self):
        """Создает тестового клиента."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Тест корневого endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_check(self, client):
        """Тест проверки здоровья."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_agent_query_endpoint(self, client):
        """Тест endpoint для запросов к агенту."""
        response = client.post(
            "/api/v1/agent/query",
            json={"query": "Покажи все продукты"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "query" in data
        assert data["query"] == "Покажи все продукты"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
