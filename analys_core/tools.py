import json
import os
import tiktoken
from ddgs import DDGS, exceptions

ARCHIVE_ROOT = "C:/Users/SKT/свалка/неон"
last_search = ""
page = 1


def execute_archive_tool(tool_call_json):
    global ARCHIVE_ROOT
    try:
        if tool_call_json[:7] == "```json":
            tool_call_json = tool_call_json[7:-3]
        tool_call = json.loads(tool_call_json)
        tool_name = tool_call.get("tool")
        path = tool_call.get("parameters").get("path")
        if path[0] != "/":
            path = "/" + path
        # --- БЕЗОПАСНОСТЬ: Проверяем, что путь находится внутри нашей корневой папки ---
        full_path = ARCHIVE_ROOT + path
        if not full_path.startswith(ARCHIVE_ROOT):
            print(full_path, "\n")
            return "TOOL_ERROR: Access denied. Path is outside of the archive root."

        # --- Выполняем команду ---
        if tool_name == "create_folder":
            os.makedirs(full_path, exist_ok=True)
            return f"TOOL_SUCCESS: Folder '{path}' created."

        elif tool_name == "write":
            content = tool_call.get("content", "")
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"TOOL_SUCCESS: File '{path}' written."

        elif tool_name == "read":
            if os.path.isfile(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif os.path.isdir(full_path):
                return json.dumps(os.listdir(full_path), ensure_ascii=False)
            else:
                return f"TOOL_ERROR: Path '{path}' not found."

        elif tool_name == "delete":
            if os.path.isfile(full_path):
                os.remove(full_path)
                return f"TOOL_SUCCESS: File '{path}' deleted."
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
                return f"TOOL_SUCCESS: Folder '{path}' deleted."
            else:
                return f"TOOL_ERROR: Path '{path}' not found to delete."

        else:
            return f"TOOL_ERROR: Unknown tool '{tool_name}'."

    except json.JSONDecodeError:
        # Это не JSON, значит, это обычное сообщение (например, финальный отчет)
        return None
    # except Exception as e:
    #     return f"TOOL_ERROR :: An unexpected error occurred: {str(e)}"


def log_it(text, num):
    try:
        with open(
            f"C:/Users/SKT/свалка/неон/logs/log{num}.txt", "r", encoding="utf-8"
        ) as f:
            temp = f.read()
    except Exception:
        temp = ""
    with open(
        f"C:/Users/SKT/свалка/неон/logs/log{num}.txt", "w", encoding="utf-8"
    ) as f:
        f.write(temp + "\n" + text)


def num_tokens_from_string(t, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = 0
    if type(t) is list:
        for i in t:
            num_tokens += len(encoding.encode(str(i)))
    elif type(t) is str:
        num_tokens = len(encoding.encode(t))
    return num_tokens


def search_this(request: str):
    try:
        global last_search
        global page
        if last_search == request:
            page += 1
            next_message = "No more information found."
        else:
            page = 1
            last_search = request
        next_message = "\n---\n".join(
            [i["body"] for i in DDGS().text(request, max_results=3, page=page)]
        )
        return next_message
    except exceptions.DDGSException:
        return "No results found."
