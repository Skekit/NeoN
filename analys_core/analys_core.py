from prompts import (
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
import google.genai as genai
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tools import search_this, execute_archive_tool, num_tokens_from_string
import os
from dotenv import load_dotenv

# Загружает переменные из файла .env (для локальной разработки)
load_dotenv("""neon\config.env""")

key1 = os.getenv("key1")
key2 = os.getenv("key2")
key3 = os.getenv("key3")
key4 = os.getenv("key4")

keys = [key1, key2, key3, key4]

model = "gemma-3-27b-it"

place_holder = ""


class agent:
    hist: list
    keys: list
    model: str
    cur_key: int
    is_temp: bool

    def __init__(
        self,
        prompt: str,
        keys: list = keys,
        model: str = model,
        is_temp: bool = False,
        is_classificate: bool = False,
        n_tryes: int = 3,
    ):
        self.hist = [
            {"role": "user", "parts": [{"text": prompt}]},
            {"role": "model", "parts": [{"text": "Aknowleged."}]},
        ]
        self.keys = keys
        self.model = model
        self.cur_key = 0
        self.is_temp = is_temp

    def send_message(self, text: str):
        self.hist.append({"role": "user", "parts": [{"text": text}]})
        while True:
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                # если история переполняется (более 15к токенов)\
                #  удаляем вторую реплику (первая - системный промпт)
                while num_tokens_from_string(self.hist) > 15_000:
                    self.hist.pop(2)
                    self.hist.pop(2)
                client = genai.Client(api_key=self.keys[self.cur_key])
                future = executor.submit(
                    client.models.generate_content, model=self.model, contents=self.hist
                )
                response = future.result(timeout=60)

                response_text = response.text
                # для временных очищаем историю
                if self.is_temp:
                    self.hist.pop(-1)
                else:
                    self.hist.append(
                        {"role": "model", "parts": [{"text": response_text}]}
                    )
                return response_text
            except TimeoutError:
                # хз почему, поэтому просто пробуем снова
                self.cur_key = (self.cur_key + 1) % len(self.keys)
                print("timeout")
                executor.shutdown(wait=False)
                time.sleep(1)
            except genai.errors.ClientError:
                # скорее всего лимит токенов, меняем ключ (там свой лимит)
                self.cur_key = (self.cur_key + 1) % len(self.keys)
                time.sleep(1)
            except Exception as e:
                err = str(e)
                print(err, end="", flush=True)
                time.sleep(10)
                print("\r" + " " * len(err) + "\r", end="", flush=True)

    def rewrite_prompt(self, new_prompt: str):
        self.hist[0] = new_prompt

    def clear_hist(self):
        self.hist = self.hist[:2]


# агенты:

Technical_Skeptik = agent(tech_skeptik)

Logical_Skeptik = agent(log_skeptik)

Consequence_Analyst = agent(con_analyst)

Pragmatist = agent(pragmatist)

Innovator = agent(innovator)

Ethicist = agent(ethicist)

Archivist = agent(archivist)

Googler = agent(googler)

# субагенты ядра:

Catalyst = agent(catalyst, is_temp=True)

Planner = agent(planner, is_temp=True)

Router = agent(router, is_temp=True)

Compressor = agent(compressor, is_temp=True)

Validator = agent(validator, is_temp=True)

ResumeNode = agent(assembler, is_temp=True)

Translator = agent(translator, is_temp=True)

Completion_Validator = agent(completion_validator_prompt, is_temp=True)

agents = {
    "Pragmatist": Pragmatist,
    "Innovator": Innovator,
    "Ethicist": Ethicist,
    "Technical_Skeptik": Technical_Skeptik,
    "Logical_Skeptik": Logical_Skeptik,
    "Consequence_Analyst": Consequence_Analyst,
    #    "Archivist": Archivist,
    "Googler": Googler,
}

available_tools = {"Search": search_this, "Archive": execute_archive_tool}


class analys_core:
    hist: str
    cur_subtask: str
    cur_iter: int
    compressed_hist: str
    original_task: str
    completed_subtasks: list
    max_iterations_per_subtask: int
    agents: dict
    available_tools: dict

    def __init__(
        self,
        max_iterations_per_subtask: int = 100,
        agents: dict = agents,
        available_tools: dict = available_tools,
    ):
        self.max_iterations_per_subtask = max_iterations_per_subtask
        self.hist = ""
        self.completed_subtasks = []
        self.compressed_hist = ""
        self.agents = agents
        self.available_tools = available_tools

    def send_task(self, task: str) -> str:
        # основной цикл решения задачи
        self.original_task = task
        self.create_subtask()
        while self.cur_subtask != "DONE":
            while True:
                next_message = self.get_next_message()
                print("Next message:\n", next_message)
                next_recipient = self.choose_recipient(next_message)
                self.get_answer(next_message, next_recipient)
                if self.valid():
                    break
            self.end_subtask()
            self.create_subtask()

        return self.end_task()

    def get_next_message(self) -> str:
        # выбор вопроса на основе истории
        return Catalyst.send_message(self.compressed_hist)

    def create_subtask(self):
        # постановка подзадачи на основе глобальной задачи и уже решенных подзадач
        self.cur_iter = 0
        message_parts = []
        if self.hist:
            message_parts.append("Story of user's dialoge:\n" + self.hist)
        if self.completed_subtasks:
            completed_text = "Completed subtasks:\n"
            for temp in self.completed_subtasks:
                completed_text += temp + "\n"
            message_parts.append(completed_text)

        # Ставим Original task В САМЫЙ КОНЕЦ, чтобы он имел максимальный вес
        message_parts.append("Original task:\n" + self.original_task)

        message = "\n\n".join(message_parts)

        verdict = Completion_Validator.send_message(message)

        if verdict == "TASK_COMPLETE":
            self.cur_subtask = "DONE"
            print("DONE")
        else:
            # Только если задача не завершена, мы планируем следующий шаг
            self.cur_subtask = Planner.send_message(message)
            self.compressed_hist += (
                "Current task:\n" + self.cur_subtask + "\n\nDiscussion history:\n"
            )
            print(f"\nGot a subtask:\n{self.cur_subtask}\n\n")

    def choose_recipient(self, next_message: str) -> str:
        # по вопросу выбираем нужного отвечающего
        message = (
            "Discussion History:\n"
            + self.compressed_hist
            + "Latest Message to be routed:\n"
            + next_message
            + "\nAvailable recipients:\n"
            + str(list(self.agents.keys()))
        )
        recipient = Router.send_message(message)
        recipient = recipient.strip()
        recipient = recipient.strip("'\"`")
        while recipient not in self.agents:
            print("Undefined recipient:", recipient)
            recipient = Router.send_message(
                message + "\nNo such agent detected:\n" + recipient
            )
            recipient = recipient.strip()
            recipient = recipient.strip("'\"`")
        return recipient

    def get_answer(self, next_message: str, next_recipient: str):
        # спрашиваем агента (авто использование инструментов и очищение истории)
        self.cur_iter += 1
        recipient = self.agents[next_recipient]
        flag = True
        while flag:
            response = recipient.send_message(next_message)
            if " :: " in response:
                if response.split("::")[0].strip() in available_tools:
                    tool = available_tools[response.split("::")[0].strip()]
                    next_message = tool(response.split("::")[1].strip())
                else:
                    flag = False
            else:
                flag = False
            print("-----")
            print(f"{next_recipient}: {response}")
            print("-----")
        self.add_to_hist(response)

    def valid(self) -> bool:
        # проверка на полноту ответа
        response = Validator.send_message(self.compressed_hist)
        if (
            response.upper() == "DONE"
            or self.cur_iter >= self.max_iterations_per_subtask
        ):
            return True
        else:
            print("-----")
            print(f"Validation recomendtions: {response}")
            self.add_to_hist(response, compress=False)
            return False

    def add_to_hist(self, text: str, compress: bool = True):
        # сжимаем и добавляем к сжатой истории
        if compress:
            compressed_text = Compressor.send_message(text)
            print("-----")
            print(f"Compressed: {compressed_text}")
            self.compressed_hist += compressed_text + "\n"
        else:
            print("\nAdded to history\n")
            self.compressed_hist += text + "\n"

    def end_subtask(self):
        # составление отчета, добавление к завершенным подзадачам и очищение истории подзадачи
        resume = ResumeNode.send_message(self.compressed_hist)
        print("-----")
        print(f"Resume for subtask: {resume}")
        self.completed_subtasks.append(f"Task: {self.cur_subtask}\nResult: {resume}")
        self.compressed_hist = ""

    def end_task(self) -> str:
        # финальный отчет, запись в глобальную историю и перевод
        message = "Original task:\n" + self.original_task
        if self.completed_subtasks:
            message += "Completed subtasks:\n"
            for temp in self.completed_subtasks:
                message += temp + "\n\n"
        final_resume = ResumeNode.send_message(message)
        self.hist += f"User:\n{self.original_task}\n\nAnswer:\n{final_resume}"
        for subject in agents.values():
            subject.clear_hist()
        return Translator.send_message(
            "User's original language message:\n"
            + self.original_task
            + "\nAnswer:\n"
            + final_resume
        )


core = analys_core()
message = input()
while message:
    print(core.send_task(message))
    message = input()
