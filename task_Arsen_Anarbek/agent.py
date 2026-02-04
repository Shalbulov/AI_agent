"""
LangGraph агент с интеграцией MCP сервера и кастомными инструментами.
"""
import json
import subprocess
import logging
from typing import Annotated, Sequence, Optional
from typing_extensions import TypedDict

# Настройка логирования
logger = logging.getLogger(__name__)

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, LLMResult


from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Состояние агента."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class MockLLM(BaseChatModel):
    """
    Мок LLM для тестирования без реальных API ключей.
    Анализирует запросы и выбирает подходящие инструменты.
    """
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatGeneration:
        """Генерирует ответ на основе сообщений."""
        last_message = messages[-1].content if messages else ""
        
        # Простой анализ запроса для выбора инструментов
        query_lower = str(last_message).lower()
        
        # Определяем, какие инструменты нужны
        if "покажи" in query_lower or "список" in query_lower or "все" in query_lower:
            if "категори" in query_lower:
                # Извлекаем категорию из запроса
                category = self._extract_category(query_lower)
                tool_calls = [{
                    "name": "mcp_list_products",
                    "args": {"category": category} if category else {}
                }]
            else:
                tool_calls = [{"name": "mcp_list_products", "args": {}}]
        elif "средняя цена" in query_lower or "статистика" in query_lower:
            tool_calls = [{"name": "mcp_get_statistics", "args": {}}]
        elif "добавь" in query_lower or "создай" in query_lower:
            # Парсим параметры продукта из запроса
            product_params = self._parse_product_params(query_lower)
            tool_calls = [{"name": "mcp_add_product", "args": product_params}]
        elif "скидк" in query_lower or "посчитай" in query_lower:
            # Для скидки сначала нужно получить продукт, затем вычислить скидку
            # Это будет сделано в два шага
            product_id, discount = self._parse_discount_params(query_lower)
            tool_calls = [
                {"name": "mcp_get_product", "args": {"product_id": product_id}}
            ]
        elif "id" in query_lower:
            # Пытаемся извлечь ID
            product_id = self._extract_id(query_lower)
            if product_id:
                tool_calls = [{"name": "mcp_get_product", "args": {"product_id": product_id}}]
            else:
                tool_calls = []
        else:
            tool_calls = []
        
        # Создаем сообщение с вызовами инструментов
        # Используем правильный формат ToolCall для LangChain
        tool_call_objects = []
        for i, tc in enumerate(tool_calls):
            tool_call_objects.append({
                "name": tc["name"],
                "args": tc["args"],
                "id": f"call_{i}"
            })
        
        message = AIMessage(content="", tool_calls=tool_call_objects)
        
        return ChatGeneration(message=message)
    
    def _extract_category(self, query: str) -> str | None:
        """Извлекает категорию из запроса."""
        categories = ["электроника", "мебель", "одежда", "продукты"]
        query_lower = query.lower()
        for cat in categories:
            if cat in query_lower:
                return cat.capitalize()
        return None
    
    def _parse_product_params(self, query: str) -> dict:
        """Парсит параметры продукта из запроса."""
        # Простой парсинг для примера
        # В реальном приложении можно использовать более сложный NLP
        params = {}
        
        # Ищем название (после "добавь" или "создай")
        if "добавь" in query.lower():
            parts = query.lower().split("добавь")[-1].split(",")
        elif "создай" in query.lower():
            parts = query.lower().split("создай")[-1].split(",")
        else:
            parts = query.split(",")
        
        for part in parts:
            part = part.strip()
            if "цена" in part or "price" in part.lower():
                try:
                    price = float(''.join(filter(str.isdigit, part)) or 0)
                    params["price"] = price
                except:
                    pass
            elif "категори" in part:
                category = part.split("категори")[-1].strip()
                params["category"] = category.capitalize()
            elif "наличие" in part or "in_stock" in part.lower():
                params["in_stock"] = "true" in part.lower() or "да" in part.lower()
            elif not params.get("name"):
                # Первое слово без ключевых слов - это название
                params["name"] = part.split()[0].capitalize() if part.split() else "Продукт"
        
        # Значения по умолчанию
        params.setdefault("name", "Новый продукт")
        params.setdefault("price", 0.0)
        params.setdefault("category", "Другое")
        params.setdefault("in_stock", True)
        
        return params
    
    def _parse_discount_params(self, query: str) -> tuple[int, float]:
        """Парсит параметры скидки из запроса."""
        # Ищем процент скидки
        discount = 15.0  # по умолчанию
        for word in query.split():
            if "%" in word:
                try:
                    discount = float(word.replace("%", ""))
                except:
                    pass
        
        # Ищем ID продукта
        product_id = 1  # по умолчанию
        if "id" in query.lower():
            parts = query.lower().split("id")
            if len(parts) > 1:
                try:
                    product_id = int(''.join(filter(str.isdigit, parts[1])))
                except:
                    pass
        
        return product_id, discount
    
    def _extract_id(self, query: str) -> int | None:
        """Извлекает ID из запроса."""
        if "id" in query.lower():
            parts = query.lower().split("id")
            if len(parts) > 1:
                try:
                    return int(''.join(filter(str.isdigit, parts[1])))
                except:
                    pass
        return None
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _stream(self, *args, **kwargs):
        raise NotImplementedError


class MCPClient:
    """Клиент для взаимодействия с MCP сервером через stdio."""
    
    def __init__(self, server_script: str = "mcp_server.py"):
        self.server_script = server_script
        self.process = None
    
    def _call_mcp_tool(self, tool_name: str, **kwargs) -> dict:
        """Вызывает инструмент MCP сервера через subprocess."""
        # Для упрощения, используем прямое обращение к функциям MCP сервера
        # В реальном приложении нужно использовать MCP протокол через stdio
        import sys
        import importlib.util
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Загружаем модуль MCP сервера
            spec = importlib.util.spec_from_file_location("mcp_server", self.server_script)
            mcp_module = importlib.util.module_from_spec(spec)
            # Не добавляем в sys.modules, чтобы избежать конфликтов
            spec.loader.exec_module(mcp_module)
            
            # Вызываем нужную функцию
            # Декоратор @mcp.tool() оборачивает функцию в FunctionTool, получаем оригинальную через .fn
            if tool_name == "list_products":
                tool_obj = getattr(mcp_module, "list_products")
                # FunctionTool имеет атрибут fn с оригинальной функцией
                if hasattr(tool_obj, "fn"):
                    func = tool_obj.fn
                else:
                    func = tool_obj
                result = func(**kwargs)
            elif tool_name == "get_product":
                tool_obj = getattr(mcp_module, "get_product")
                if hasattr(tool_obj, "fn"):
                    func = tool_obj.fn
                else:
                    func = tool_obj
                result = func(**kwargs)
            elif tool_name == "add_product":
                tool_obj = getattr(mcp_module, "add_product")
                if hasattr(tool_obj, "fn"):
                    func = tool_obj.fn
                else:
                    func = tool_obj
                result = func(**kwargs)
            elif tool_name == "get_statistics":
                tool_obj = getattr(mcp_module, "get_statistics")
                if hasattr(tool_obj, "fn"):
                    func = tool_obj.fn
                else:
                    func = tool_obj
                result = func(**kwargs)
            else:
                raise ValueError(f"Неизвестный инструмент: {tool_name}")
            
            logger.info(f"MCP tool {tool_name} вызван успешно, результат: {type(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при вызове MCP tool {tool_name}: {str(e)}", exc_info=True)
            raise


# Инициализация MCP клиента
mcp_client = MCPClient()


# Кастомные инструменты агента
@tool
def calculate_discount(price: float, discount_percent: float) -> dict:
    """
    Вычисляет цену со скидкой.
    
    Args:
        price: Исходная цена
        discount_percent: Процент скидки
        
    Returns:
        Словарь с исходной ценой, процентом скидки, суммой скидки и итоговой ценой
    """
    discount_amount = price * (discount_percent / 100)
    final_price = price - discount_amount
    
    return {
        "original_price": price,
        "discount_percent": discount_percent,
        "discount_amount": round(discount_amount, 2),
        "final_price": round(final_price, 2)
    }


@tool
def format_currency(amount: float, currency: str = "RUB") -> str:
    """
    Форматирует сумму в валютном формате.
    
    Args:
        amount: Сумма
        currency: Валюта (по умолчанию RUB)
        
    Returns:
        Отформатированная строка с валютой
    """
    currency_symbols = {
        "RUB": "₽",
        "USD": "$",
        "EUR": "€"
    }
    symbol = currency_symbols.get(currency, currency)
    return f"{amount:,.2f} {symbol}"


from pydantic import BaseModel, Field

# Схемы аргументов для инструментов
class ListProductsSchema(BaseModel):
    category: Optional[str] = Field(None, description="Опциональная категория для фильтрации продуктов")

class GetProductSchema(BaseModel):
    product_id: int = Field(..., description="ID продукта")

class AddProductSchema(BaseModel):
    name: str = Field(..., description="Название продукта")
    price: float = Field(..., description="Цена продукта")
    category: str = Field(..., description="Категория продукта")
    in_stock: bool = Field(True, description="Наличие на складе")

class GetStatisticsSchema(BaseModel):
    pass

# Создаем обертки для MCP инструментов
def create_mcp_tool(tool_name: str, args_schema: type[BaseModel]):
    """Создает LangChain tool из MCP инструмента."""
    def mcp_tool_wrapper(**kwargs):
        """Обертка для MCP инструмента."""
        try:
            result = mcp_client._call_mcp_tool(tool_name, **kwargs)
            # Возвращаем результат как JSON строку для правильной обработки
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    # Определяем docstring и имя
    if tool_name == "list_products":
        description = "Получает список продуктов (опционально по категории)"
    elif tool_name == "get_product":
        description = "Получает продукт по ID"
    elif tool_name == "add_product":
        description = "Добавляет новый продукт"
    elif tool_name == "get_statistics":
        description = "Получает статистику о продуктах"
    else:
        description = f"MCP инструмент: {tool_name}"
    
    # Создаем инструмент с кастомным именем и схемой
    return StructuredTool.from_function(
        func=mcp_tool_wrapper,
        name=f"mcp_{tool_name}",
        description=description,
        args_schema=args_schema
    )


# Создаем все инструменты
mcp_list_products = create_mcp_tool("list_products", ListProductsSchema)
mcp_get_product = create_mcp_tool("get_product", GetProductSchema)
mcp_add_product = create_mcp_tool("add_product", AddProductSchema)
mcp_get_statistics = create_mcp_tool("get_statistics", GetStatisticsSchema)

# Все доступные инструменты
tools = [
    mcp_list_products,
    mcp_get_product,
    mcp_add_product,
    mcp_get_statistics,
    calculate_discount,
    format_currency
]


def should_continue(state: AgentState) -> str:
    """Определяет, нужно ли продолжать выполнение."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Если последнее сообщение - вызов инструмента, продолжаем
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    
    return "end"


def call_model(state: AgentState) -> AgentState:
    """Вызывает модель для обработки запроса."""
    messages = state["messages"]
    llm = MockLLM()
    
    # Если есть результаты инструментов, проверяем, нужно ли вызывать еще инструменты
    if len(messages) > 1:
        last_tool_message = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break
        
        if last_tool_message:
            # Проверяем, был ли это запрос на скидку
            original_query = messages[0].content if messages else ""
            if ("скидк" in original_query.lower() or "посчитай" in original_query.lower()) and "mcp_get_product" in str(last_tool_message.name):
                # Получаем цену продукта и вычисляем скидку
                try:
                    product_data = json.loads(last_tool_message.content)
                    price = product_data.get("price", 0)
                    # Парсим процент скидки из оригинального запроса
                    discount = 15.0  # по умолчанию
                    for word in original_query.lower().split():
                        if "%" in word:
                            try:
                                discount = float(word.replace("%", ""))
                            except:
                                pass
                    # Вызываем инструмент расчета скидки
                    tool_calls = [{
                        "name": "calculate_discount",
                        "args": {"price": price, "discount_percent": discount},
                        "id": "call_discount"
                    }]
                    response = AIMessage(content="", tool_calls=tool_calls)
                    return {"messages": [response]}
                except Exception as e:
                    # Если ошибка, просто формируем ответ из результата продукта
                    pass
            
            # Формируем финальный ответ на основе результатов инструментов
            tool_results = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    try:
                        if _is_json(msg.content):
                            tool_results.append(_format_tool_result(json.loads(msg.content)))
                        else:
                            tool_results.append(msg.content)
                    except:
                        tool_results.append(msg.content)
            
            if tool_results:
                response_text = "\n".join([tr for tr in tool_results if tr])
                response = AIMessage(content=response_text)
                return {"messages": [response]}
            else:
                # Если нет результатов, возвращаем пустое сообщение
                response = AIMessage(content="Не удалось получить результаты.")
                return {"messages": [response]}
    
    # Анализируем запрос и выбираем инструменты
    # MockLLM не поддерживает bind_tools, поэтому вызываем _generate напрямую
    generation = llm._generate(messages)
    response = generation.message
    
    return {"messages": [response]}


def call_tools(state: AgentState) -> AgentState:
    """Вызывает инструменты на основе tool_calls."""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_messages = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_node = ToolNode(tools)
        tool_results = tool_node.invoke({"messages": [last_message]})
        tool_messages = tool_results["messages"]
    
    return {"messages": tool_messages}


# Создание графа агента
def create_agent():
    """Создает и возвращает LangGraph агента."""
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    
    # Добавляем ребра
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# Глобальный экземпляр агента
agent = create_agent()


def process_query(query: str) -> str:
    """
    Обрабатывает запрос пользователя через агента.
    
    Args:
        query: Запрос пользователя
        
    Returns:
        Ответ агента
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Создаем начальное состояние
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Запускаем агента
        result = agent.invoke(initial_state)
        
        # Извлекаем ответ
        messages = result["messages"]
        logger.info(f"Получено {len(messages)} сообщений от агента")
        
        # Ищем последнее сообщение от агента или инструмента
        response_parts = []
        
        # Сначала собираем все ToolMessage
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        for msg in tool_messages:
            try:
                # Пытаемся распарсить JSON
                tool_result = json.loads(msg.content)
                formatted = _format_tool_result(tool_result)
                if formatted:
                    response_parts.append(formatted)
            except (json.JSONDecodeError, TypeError):
                # Если не JSON, используем как есть
                if msg.content and msg.content.strip():
                    response_parts.append(msg.content)
        
        # Затем ищем AIMessage с контентом
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                # Пропускаем пустые сообщения с только tool_calls
                if not (hasattr(msg, "tool_calls") and msg.tool_calls and not msg.content):
                    response_parts.append(msg.content)
        
        # Если все еще нет ответа, используем последний ToolMessage
        if not response_parts and tool_messages:
            last_tool = tool_messages[-1]
            try:
                tool_result = json.loads(last_tool.content)
                response_parts.append(_format_tool_result(tool_result))
            except:
                response_parts.append(last_tool.content)
        
        result_text = "\n".join(response_parts) if response_parts else "Не удалось обработать запрос."
        logger.info(f"Сформирован ответ: {result_text[:100]}...")
        return result_text
        
    except Exception as e:
        logger.error(f"Ошибка в process_query: {str(e)}", exc_info=True)
        return f"Произошла ошибка при обработке запроса: {str(e)}"


def _is_json(s: str) -> bool:
    """Проверяет, является ли строка валидным JSON."""
    try:
        json.loads(s)
        return True
    except:
        return False


def _format_tool_result(result) -> str:
    """Форматирует результат инструмента для пользователя."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            return result
    
    if isinstance(result, list):
        if not result:
            return "Продукты не найдены."
        formatted = []
        for p in result:
            in_stock = "✅ В наличии" if p.get('in_stock', False) else "❌ Нет в наличии"
            formatted.append(
                f"📦 {p.get('name')}\n"
                f"   ID: {p.get('id')}\n"
                f"   Цена: {p.get('price')}₽\n"
                f"   Категория: {p.get('category')}\n"
                f"   {in_stock}"
            )
        return "\n\n".join(formatted)
    elif isinstance(result, dict):
        if "count" in result:
            return f"📊 Статистика:\n   Количество продуктов: {result['count']}\n   Средняя цена: {result['average_price']}₽"
        elif "final_price" in result:
            return (
                f"💰 Расчет скидки:\n"
                f"   Исходная цена: {result['original_price']}₽\n"
                f"   Скидка: {result['discount_percent']}%\n"
                f"   Сумма скидки: {result['discount_amount']}₽\n"
                f"   Итоговая цена: {result['final_price']}₽"
            )
        elif "id" in result:
            in_stock = "✅ В наличии" if result.get('in_stock', False) else "❌ Нет в наличии"
            return (
                f"📦 {result.get('name')}\n"
                f"   ID: {result.get('id')}\n"
                f"   Цена: {result.get('price')}₽\n"
                f"   Категория: {result.get('category')}\n"
                f"   {in_stock}"
            )
    return str(result)
