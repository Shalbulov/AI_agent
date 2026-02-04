#!/bin/bash

# Скрипт для тестирования API

echo "=== Тестирование AI Agent API ==="
echo ""

# Проверка здоровья
echo "1. Health Check:"
curl -s http://localhost:8080/health | jq '.' || curl -s http://localhost:8080/health
echo ""
echo ""

# Корневой endpoint
echo "2. Root endpoint:"
curl -s http://localhost:8080/ | jq '.' || curl -s http://localhost:8080/
echo ""
echo ""

# Тестовый запрос - список продуктов
echo "3. Запрос: Покажи все продукты"
curl -s -X POST "http://localhost:8080/api/v1/agent/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Покажи все продукты"}' | jq '.' || \
curl -s -X POST "http://localhost:8080/api/v1/agent/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Покажи все продукты"}'
echo ""
echo ""

# Запрос статистики
echo "4. Запрос: Какая средняя цена продуктов?"
curl -s -X POST "http://localhost:8080/api/v1/agent/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Какая средняя цена продуктов?"}' | jq '.' || \
curl -s -X POST "http://localhost:8080/api/v1/agent/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Какая средняя цена продуктов?"}'
echo ""
echo ""

echo "=== Тестирование завершено ==="
