"""
MCP Server для управления продуктами.
Использует FastMCP для создания сервера, работающего через stdio.
"""
import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Optional

from fastmcp import FastMCP

# Инициализация FastMCP сервера
mcp = FastMCP("Product Management Server")

# Путь к файлу базы данных
DB_FILE = Path("products.db")


def get_db_connection():
    """Создает подключение к базе данных."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Инициализирует базу данных."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Создаем таблицу, если её нет
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL,
            in_stock BOOLEAN NOT NULL DEFAULT 1
        )
    ''')
    
    # Проверяем, пуста ли таблица
    cursor.execute('SELECT count(*) FROM products')
    if cursor.fetchone()[0] == 0:
        # Добавляем начальные данные
        initial_products = [
            ("Ноутбук", 50000, "Электроника", True),
            ("Смартфон", 30000, "Электроника", True),
            ("Стол", 8000, "Мебель", False)
        ]
        cursor.executemany(
            'INSERT INTO products (name, price, category, in_stock) VALUES (?, ?, ?, ?)',
            initial_products
        )
        conn.commit()
        print("База данных инициализирована начальными данными.")
    
    conn.commit()
    conn.close()


@mcp.tool()
def list_products(category: Optional[str] = None) -> List[Dict]:
    """
    Получает список всех продуктов или продуктов в указанной категории.
    
    Args:
        category: Опциональная категория для фильтрации продуктов
        
    Returns:
        Список словарей с информацией о продуктах
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if category:
        cursor.execute('SELECT * FROM products WHERE category = ?', (category.capitalize(),)) # Capitalize to match logic roughly
    else:
        cursor.execute('SELECT * FROM products')
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


@mcp.tool()
def get_product(product_id: int) -> Dict:
    """
    Получает информацию о продукте по его ID.
    
    Args:
        product_id: ID продукта
        
    Returns:
        Словарь с информацией о продукте
        
    Raises:
        ValueError: Если продукт с указанным ID не найден
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    else:
        raise ValueError(f"Продукт с ID {product_id} не найден")


@mcp.tool()
def add_product(name: str, price: float, category: str, in_stock: bool = True) -> Dict:
    """
    Добавляет новый продукт в базу данных.
    
    Args:
        name: Название продукта
        price: Цена продукта
        category: Категория продукта
        in_stock: Наличие на складе (по умолчанию True)
        
    Returns:
        Словарь с информацией о добавленном продукте
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO products (name, price, category, in_stock) VALUES (?, ?, ?, ?)',
        (name, price, category, in_stock)
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    
    return {
        "id": new_id,
        "name": name,
        "price": price,
        "category": category,
        "in_stock": in_stock
    }


@mcp.tool()
def get_statistics() -> Dict:
    """
    Получает статистику о продуктах: количество и средняя цена.
    
    Returns:
        Словарь со статистикой: count (количество) и average_price (средняя цена)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*), AVG(price) FROM products')
    row = cursor.fetchone()
    conn.close()
    
    count = row[0]
    average_price = row[1] if row[1] is not None else 0.0
    
    return {
        "count": count,
        "average_price": round(average_price, 2)
    }



# Инициализация базы данных при импорте модуля
init_db()

if __name__ == "__main__":
    # Запуск MCP сервера через stdio
    mcp.run(transport="stdio")
