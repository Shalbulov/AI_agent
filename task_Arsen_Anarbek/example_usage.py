"""
Пример использования API агента.
"""
import requests
import json

BASE_URL = "http://localhost:8080"


def test_queries():
    """Тестирует различные запросы к агенту."""
    
    queries = [
        "Покажи все продукты",
        "Покажи все продукты в категории Электроника",
        "Какая средняя цена продуктов?",
        "Добавь новый продукт: Мышка, цена 1500, категория Электроника",
        "Посчитай скидку 15% на товар с ID 1"
    ]
    
    print("=" * 60)
    print("Примеры запросов к AI агенту")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Запрос: {query}")
        print("-" * 60)
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/agent/query",
                json={"query": query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Ответ: {data['response']}")
            else:
                print(f"Ошибка: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            print("Ошибка: Не удалось подключиться к серверу.")
            print("Убедитесь, что сервер запущен: docker-compose up")
            break
        except Exception as e:
            print(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    test_queries()
