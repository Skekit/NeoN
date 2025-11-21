"""
Гибридная мульти-агентная система с асинхронной доской объявлений
ИСПРАВЛЕНА ТИПИЗАЦИЯ ДЛЯ MyPy
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Sequence, Union, cast
from collections import deque
from itertools import cycle
import time

# Внешние зависимости
import google.genai as genai
import google.genai.errors as genai_errors
from google.genai.types import ContentDict, PartDict

# Локальные модули
from test_prompts import (
    tech_skeptik,
    con_analyst,
    log_skeptik,
    googler,
    pragmatist,
    planner,
    validator,
    router,
    translator,
    assembler,
    archivist,
    innovator,
    ethicist,
    catalyst,
    compressor,
    completion_validator_prompt,
)
from tools import search_this, execute_archive_tool, num_tokens_from_string

# ============ КОНФИГУРАЦИЯ (ЗАПОЛНИТЕ СВОИ ЗНАЧЕНИЯ!) ============
API_KEYS: List[str] = [
    "AIzaSyAfPK51eeunFf5x43595zwZTOx7JH-mdhA",
    "AIzaSyDvLf0xvq1KLiNMwi6rHw74I-AB2u8ewMk",
    "AIzaSyD1FPjyHlYnydiUbfyOs-FaihHiDnGjdV0",
    "AIzaSyBBzBx-R4gPDOTiBuWGbHvayCFiC4_tEn0",
]
MODEL_NAME: str = "gemma-3-27b-it"
MAX_TOKENS_PER_AGENT: int = 15000
RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
# ===================================================================


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============


def num_tokens_from_text(text: str) -> int:
    """Подсчет токенов в тексте"""
    return len(text) // 4


# ============ BLACKBOARD СТРУКТУРЫ ============


@dataclass
class Message:
    """Сообщение на доске объявлений"""

    sender: str
    content: str
    topic: str = "general"
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def format_for_context(self) -> str:
        return f"[{self.sender}] {self.content}"


# ============ АСИНХРОННЫЙ АГЕНТ ============


class AsyncAgent:
    """Асинхронный агент с правильной историей и защитой от ошибок"""

    def __init__(
        self,
        system_prompt: str,
        api_keys: List[str],
        model: str = MODEL_NAME,
        is_temp: bool = False,
        name: str = "UnnamedAgent",
    ):
        self.name: str = name
        self._hist: List[Dict[str, Union[str, List[PartDict]]]] = [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "model", "parts": [{"text": "Acknowledged."}]},
        ]
        self._api_keys = cycle(api_keys)
        self._model: str = model
        self._is_temp: bool = is_temp
        self._semaphore = asyncio.Semaphore(
            RATE_LIMIT_REQUESTS_PER_MINUTE // len(api_keys)
        )
        self._current_key: str = next(self._api_keys)
        logger.info(f"Initialized agent {self.name} with model {self.model}")

    @property
    def model(self) -> str:
        return self._model

    async def send_message(self, text: str, max_retries: int = 3) -> str:
        """
        Асинхронная отправка сообщения с правильной обработкой истории,
        ротацией ключей и retry с экспоненциальной задержкой
        """
        # Удаляем старые сообщения, если превышен лимит токенов
        while (
            len(self._hist) > 2
            and num_tokens_from_string(
                cast(List[Dict[str, List[PartDict]]], self._hist)
            )
            > MAX_TOKENS_PER_AGENT
        ):
            logger.warning(f"{self.name}: Trimming history due to token limit")
            self._hist.pop(2)  # Удаляем user message
            if len(self._hist) > 2:
                self._hist.pop(2)  # Удаляем model response

        self._hist.append({"role": "user", "parts": [{"text": text}]})

        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    client = genai.Client(api_key=self._current_key)

                    # Выполняем в thread pool, чтобы не блокировать event loop
                    loop = asyncio.get_event_loop()
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: client.models.generate_content(
                                model=self._model, contents=str(self._hist)
                            ),
                        ),
                        timeout=60.0,
                    )

                    response_text = response.text if response.text else ""

                    if self._is_temp:
                        # Для временных агентов не сохраняем ответ в историю
                        self._hist.pop()  # Удаляем последний user message
                    else:
                        self._hist.append(
                            {"role": "model", "parts": [{"text": response_text}]}
                        )

                    logger.debug(
                        f"{self.name}: Generated response ({len(response_text)} chars)"
                    )
                    return response_text

                except asyncio.TimeoutError:
                    logger.warning(
                        f"{self.name}: Timeout, rotating key and retrying..."
                    )
                    self._rotate_key()
                    await asyncio.sleep(2**attempt)  # Экспоненциальная задержка

                except genai_errors.ClientError as e:
                    logger.error(f"{self.name}: API client error: {e}")
                    self._rotate_key()
                    await asyncio.sleep(2**attempt)

                except Exception as e:
                    logger.error(f"{self.name}: Unexpected error: {e}", exc_info=True)
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed after {max_retries} attempts: {e}")
                    await asyncio.sleep(5 * (attempt + 1))

        # MyPy требует явного return вне цикла
        raise RuntimeError(f"Failed to send message after {max_retries} attempts")

    def _rotate_key(self) -> None:
        """Ротация API ключа"""
        self._current_key = next(self._api_keys)
        logger.info(f"{self.name}: Rotated API key")

    def rewrite_prompt(self, new_prompt: str) -> None:
        """Безопасная замена системного промпта"""
        if len(self._hist) >= 1:
            self._hist[0] = {"role": "user", "parts": [{"text": new_prompt}]}
        else:
            raise ValueError("History is too short to rewrite prompt")

    def clear_hist(self) -> None:
        """Очищаем историю, сохраняя только системный промпт"""
        if len(self._hist) >= 2:
            self._hist = self._hist[:2]
        else:
            self._hist = self._hist[:1] if self._hist else []
        logger.info(f"{self.name}: Cleared conversation history")

    async def estimate_relevance(self, topic: str, content: str) -> float:
        """Оценка релевантности агента для конкретной темы (для доски)"""
        relevance_prompt = f"""
        Rate how relevant you are to this topic (0-1):
        Topic: {topic}
        Content: {content}
        Your expertise: {self.name}
        
        Respond with only a number between 0 and 1.
        """
        try:
            response = await self.send_message(relevance_prompt, max_retries=1)
            return float(response.strip())
        except (ValueError, RuntimeError):
            return 0.5

    async def estimate_confidence(self, content: str) -> float:
        """Оценка уверенности в ответе (для приоритета)"""
        confidence_prompt = f"""
        Rate your confidence in this response (0-1):
        {content}
        
        Respond with only a number between 0 and 1.
        """
        try:
            response = await self.send_message(confidence_prompt, max_retries=1)
            return float(response.strip())
        except (ValueError, RuntimeError):
            return 0.5


# ============ BLACKBOARD ============


class Blackboard:
    """
    Дискуссионная доска с аукционом агентов
    Агенты сами решают, когда и что отвечать
    """

    def __init__(self, agents: Dict[str, AsyncAgent], tools: Dict[str, Callable]):
        self.agents: Dict[str, AsyncAgent] = agents
        self.tools: Dict[str, Callable] = tools
        self.board: List[Message] = []
        self.lock = asyncio.Lock()
        self._message_event = asyncio.Event()

        logger.info(
            f"Blackboard initialized with {len(agents)} agents and {len(tools)} tools"
        )

    async def post_message(self, msg: Message) -> None:
        """Разместить сообщение и уведомить всех агентов"""
        async with self.lock:
            self.board.append(msg)
            logger.info(
                f"[BOARD] {msg.sender} -> {msg.topic} (priority: {msg.priority:.2f})"
            )

        self._message_event.set()
        self._message_event.clear()

    async def get_recent_context(self, last_n: int = 10) -> str:
        """Получить последние сообщения для контекста"""
        async with self.lock:
            recent = self.board[-last_n:]
            return "\n\n".join([msg.format_for_context() for msg in recent])

    async def get_relevant_agents(
        self, msg: Message, top_k: int = 3
    ) -> List[tuple[str, float]]:
        """
        Аукцион: какие агенты наиболее релевантны для этого сообщения?
        """
        tasks: List[tuple[str, asyncio.Task[float]]] = []
        for agent_name, agent in self.agents.items():
            if agent_name == msg.sender:
                continue
            task = asyncio.create_task(agent.estimate_relevance(msg.topic, msg.content))
            tasks.append((agent_name, task))

        # Ждем всех
        results: List[tuple[str, float]] = []
        for agent_name, task in tasks:
            try:
                relevance = await task
                if relevance > 0.3:
                    results.append((agent_name, relevance))
            except Exception as e:
                logger.warning(f"Failed to estimate relevance for {agent_name}: {e}")

        # Сортируем
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def execute_tool_if_needed(self, response_text: str) -> Optional[str]:
        """Обработка вызова инструмента"""
        if " :: " not in response_text:
            return None

        tool_part, input_part = response_text.split(" :: ", 1)
        tool_name = tool_part.strip()
        tool_input = input_part.strip()

        if tool_name in self.tools:
            logger.info(
                f"[TOOL] Executing {tool_name} with input: {tool_input[:50]}..."
            )
            try:
                result = self.tools[tool_name](tool_input)
                logger.info(f"[TOOL] {tool_name} completed")
                return str(result)  # Гарантируем возврат строки
            except Exception as e:
                logger.error(f"[TOOL] {tool_name} failed: {e}")
                return f"Tool error: {e}"

        return None

    async def run_discussion(
        self, initial_task: str, max_rounds: int = 8, min_agents_per_round: int = 2
    ) -> List[Message]:
        """Запуск дискуссии на доске"""

        await self.post_message(
            Message(
                sender="Human",
                content=initial_task,
                topic="task_definition",
                priority=1.0,
            )
        )

        for round_num in range(max_rounds):
            logger.info(f"\n{'='*20} Round {round_num + 1} {'='*20}")

            async with self.lock:
                last_message = self.board[-1] if self.board else None

            if not last_message:
                break

            # Находим релевантных агентов
            relevant_agents = await self.get_relevant_agents(last_message)

            if len(relevant_agents) < min_agents_per_round:
                logger.info(
                    f"Only {len(relevant_agents)} agents want to respond, ending discussion"
                )
                break

            logger.info(f"Relevant agents: {[name for name, _ in relevant_agents]}")

            # Запускаем параллельно
            tasks = []
            for agent_name, relevance in relevant_agents:
                agent = self.agents[agent_name]
                task = self._agent_respond(agent_name, agent, last_message)
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Обрабатываем ошибки
            for i, result in enumerate(responses):
                if isinstance(result, Exception):
                    agent_name = relevant_agents[i][0]
                    logger.error(f"Agent {agent_name} failed: {result}")

            # Проверяем консенсус
            if await self._check_consensus():
                logger.info("Consensus reached, ending discussion")
                break

        return self.board

    async def _agent_respond(
        self, agent_name: str, agent: AsyncAgent, trigger_msg: Message
    ) -> None:
        """Один агент формирует ответ"""
        try:
            context = await self.get_recent_context(last_n=5)

            response_text = await agent.send_message(
                f"Discussion context:\n{context}\n\n"
                f"Respond to: {trigger_msg.sender}: {trigger_msg.content}"
            )

            # Обработка инструмента
            tool_result = await self.execute_tool_if_needed(response_text)
            if tool_result:
                response_text = await agent.send_message(
                    f"Tool result: {tool_result}\nOriginal question: {trigger_msg.content}"
                )

            # Оценка приоритета
            priority = await agent.estimate_confidence(response_text)

            await self.post_message(
                Message(
                    sender=agent_name,
                    content=response_text,
                    topic="general",
                    priority=priority,
                )
            )

        except Exception as e:
            logger.error(f"Error in agent {agent_name}: {e}")
            raise

    async def _check_consensus(self) -> bool:
        """Проверяем, завершена ли дискуссия"""
        validator = self.agents.get("Validator")
        if not validator:
            return False

        try:
            context = await self.get_recent_context(last_n=10)
            verdict = await validator.send_message(
                f"Is this discussion complete and coherent? Rate 0-1.\n\n{context}"
            )

            # Парсим число
            try:
                score = float(verdict.strip().split()[0])
                return score > 0.85
            except (ValueError, IndexError):
                return "DONE" in verdict.upper() or "COMPLETE" in verdict.upper()

        except Exception as e:
            logger.error(f"Consensus check failed: {e}")
            return False


# ============ HYBRID CORE ============


class HybridCore:
    """
    Гибридное ядро: планируем каскадом, обсуждаем доской, собираем результат
    """

    def __init__(
        self,
        agents: Dict[str, AsyncAgent],
        tools: Dict[str, Callable],
        max_iterations: int = 100,
    ):
        self.agents: Dict[str, AsyncAgent] = agents
        self.tools: Dict[str, Callable] = tools
        self.max_iterations: int = max_iterations
        self.global_history: List[str] = []
        self.completed_subtasks: List[Dict[str, str]] = []

        logger.info(f"HybridCore initialized with {len(agents)} agents")

    async def send_task(self, original_task: str) -> str:
        """Основной метод для решения задачи"""

        logger.info(f"\n{'='*50}")
        logger.info(f"NEW TASK: {original_task[:100]}...")
        logger.info(f"{'='*50}\n")

        # Фаза 1: Проверка завершенности
        completion_validator = self.agents.get("Completion_Validator")
        if completion_validator:
            try:
                verdict = await completion_validator.send_message(
                    f"Task: {original_task}\nCheck if already complete."
                )
                if "COMPLETE" in verdict.upper():
                    logger.info("Task marked as complete by validator")
                    return await self._finalize_task(original_task)
            except Exception as e:
                logger.warning(f"Completion validation failed: {e}")

        # Фаза 2: Планирование подзадач
        planner = self.agents.get("Planner")
        if not planner:
            raise ValueError("Planner agent not found")

        completed_summary = "\n".join(
            [f"- {st['task'][:50]}..." for st in self.completed_subtasks[-5:]]
        )

        planning_msg = f"Original task:\n{original_task}\n"
        if completed_summary:
            planning_msg += f"Completed:\n{completed_summary}\n"

        subtask = await planner.send_message(planning_msg)

        if (
            "DONE" in subtask.upper()
            or len(self.completed_subtasks) >= self.max_iterations
        ):
            logger.info("No more subtasks, finalizing")
            return await self._finalize_task(original_task)

        logger.info(f"\n[PLANNER] New subtask: {subtask[:100]}...")

        # Фаза 3: Дискуссия
        discussion_agents = {
            name: agent
            for name, agent in self.agents.items()
            if name
            not in ["Planner", "Completion_Validator", "Assembler", "Translator"]
        }

        blackboard = Blackboard(discussion_agents, self.tools)
        messages = await blackboard.run_discussion(initial_task=subtask, max_rounds=6)

        # Фаза 4: Сборка результата
        discussion_text = "\n\n".join([msg.format_for_context() for msg in messages])

        assembler = self.agents.get("Assembler")
        if not assembler:
            raise ValueError("Assembler agent not found")

        subtask_result = await assembler.send_message(
            f"Subtask: {subtask}\n\nDiscussion:\n{discussion_text}"
        )

        self.completed_subtasks.append(
            {"task": subtask, "result": subtask_result, "discussion": discussion_text}
        )

        logger.info(f"[ASSEMBLER] Subtask result: {subtask_result[:100]}...")

        # Рекурсивно продолжаем
        return await self.send_task(original_task)

    async def _finalize_task(self, original_task: str) -> str:
        """Финальная сборка и перевод"""

        logger.info("\n" + "=" * 20 + " FINALIZING " + "=" * 20)

        if not self.completed_subtasks:
            return "No results to assemble."

        all_results = "\n\n".join(
            [
                f"## {st['task'][:50]}...\n{st['result']}"
                for st in self.completed_subtasks
            ]
        )

        assembler = self.agents.get("Assembler")
        if not assembler:
            raise ValueError("Assembler agent not found")

        final_answer = await assembler.send_message(
            f"Original task: {original_task}\n\nResults:\n{all_results}"
        )

        # Перевод
        translator = self.agents.get("Translator")
        if translator:
            translated = await translator.send_message(
                f"Translate to user's language:\n{final_answer}"
            )
            final_answer = translated or final_answer

        # Очищаем истории
        await self._clear_all_agents()

        # Сохраняем глобальную историю
        self.global_history.append(
            f"User: {original_task}\nAssistant: {final_answer[:200]}..."
        )

        # Сброс для следующей задачи
        self.completed_subtasks = []

        logger.info("Task completed successfully!")
        return final_answer

    async def _clear_all_agents(self) -> None:
        """Очищаем историю всех агентов"""
        tasks = [asyncio.to_thread(agent.clear_hist) for agent in self.agents.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Cleared history for all agents")

    async def _get_cached_result(self, original_task: str) -> str:
        """Возвращаем кэшированный результат"""
        if self.completed_subtasks:
            return self.completed_subtasks[-1]["result"]
        return "Task marked complete, but no cached result."


# ============ ФАБРИКА АГЕНТОВ ============


def create_agents(api_keys: List[str]) -> Dict[str, AsyncAgent]:
    """Создает все агентов с правильно типизированными промптами"""

    # Явно указываем типы для MyPy
    agent_configs: Dict[str, Dict[str, Any]] = {
        "Technical_Skeptik": {"prompt": str(tech_skeptik), "is_temp": False},
        "Logical_Skeptik": {"prompt": str(log_skeptik), "is_temp": False},
        "Consequence_Analyst": {"prompt": str(con_analyst), "is_temp": False},
        "Pragmatist": {"prompt": str(pragmatist), "is_temp": False},
        "Innovator": {"prompt": str(innovator), "is_temp": False},
        "Ethicist": {"prompt": str(ethicist), "is_temp": False},
        "Archivist": {"prompt": str(archivist), "is_temp": False},
        "Googler": {"prompt": str(googler), "is_temp": False},
        # Управляющие агенты
        "Catalyst": {"prompt": str(catalyst), "is_temp": True},
        "Planner": {"prompt": str(planner), "is_temp": True},
        "Router": {"prompt": str(router), "is_temp": True},
        "Compressor": {"prompt": str(compressor), "is_temp": True},
        "Validator": {"prompt": str(validator), "is_temp": True},
        "Assembler": {"prompt": str(assembler), "is_temp": True},
        "Translator": {"prompt": str(translator), "is_temp": True},
        "Completion_Validator": {
            "prompt": str(completion_validator_prompt),
            "is_temp": True,
        },
    }

    agents: Dict[str, AsyncAgent] = {}
    for name, config in agent_configs.items():
        agents[name] = AsyncAgent(
            system_prompt=str(config["prompt"]),
            api_keys=api_keys,
            model=MODEL_NAME,
            is_temp=bool(config["is_temp"]),
            name=name,
        )

    return agents


# ============ ИНСТРУМЕНТЫ ============


def get_tools() -> Dict[str, Callable]:
    """Возвращает доступные инструменты"""
    return {
        "Search": search_this,
        "Archive": execute_archive_tool,
    }


# ============ MAIN ============


async def interactive_mode():
    """Интерактивный режим"""
    logger.info("Initializing hybrid multi-agent system...")

    if not API_KEYS or "YOUR_API_KEY" in API_KEYS[0]:
        logger.error("Please configure your API keys in the API_KEYS variable!")
        return

    agents = create_agents(API_KEYS)
    tools = get_tools()
    core = HybridCore(agents=agents, tools=tools, max_iterations=100)

    print("\n" + "=" * 60)
    print("Hybrid Multi-Agent Assistant Ready (MyPy Compliant)")
    print("Enter your task (empty line to quit)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
            if not user_input.strip():
                logger.info("Shutting down...")
                break

            response = await core.send_task(user_input)

            print("\n" + "-" * 60)
            print(f"Assistant: {response}")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            print(f"\nError: {e}\n")
            continue


try:
    asyncio.run(interactive_mode())
except KeyboardInterrupt:
    logger.info("Application terminated")

