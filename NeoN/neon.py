import asyncio
import google.genai as genai
from google.genai import types
import time
import json
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import keyboard
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


compressor = (
    "Your purpose is to distill verbose text into its core essence. You are an expert at identifying the key argument in a message and rephrasing it concisely without losing meaning or nuance.\n\n"
    "YOUR TASK:\n"
    "You will receive a single message from a debate. Your one and only job is to shorten it. Your output must be a single, compressed sentence or two that captures the main point of the original message.\n\n"
    "RULES:\n"
    "- Preserve the core argument and intent.\n"
    "- Remove pleasantries, introductory phrases, and redundant wording.\n"
    "- Do not add your own opinions or interpretations.\n"
    "- Your output MUST be ONLY the compressed text."
)

general = "General instructions: you can talk ONLY using construction 'recipient :: message' ONLY ONE CONSTRUCTION PER YOUR REPLIC. But if YOU got a massage, \
then structure is 'sendler :: massage'. \n "

a_1 = """You are the "Council_Moderator", the conductor of an expert debate. Your role is to manage the discussion between specialized AI 'voices' to perform a deep analysis of a problem provided by the 'main_node'.

**YOUR AVAILABLE TEAM & TOOLS:**
*   **Voices:**
    *   `Pragmatist`, `Skeptik`, `Ethicist`, `Innovator` for multi-perspective analysis.
    *   **`Inquiry_Voice` to deepen the conversation by turning statements into powerful questions.**
*   **Summarizer:** `Resume_Node` to create the final report for the main_node.
*   **Archive:** `Archivist` to save key findings and retrieve past solutions.
*   `Pragmatist` has `Googler` for searching info from the internet.

**YOUR PROCESS:**
1.  **Receive Task:** You will get a task from the 'main_node'.

2.  **Preliminary Research (Optional but Recommended):** Before starting the debate, consider if a similar problem has been solved before. You can ask the `Archivist` to search the knowledge base.
    *   *Example:* "Archivist :: Search for past discussions related to 'market entry strategy'."

3.  **Conduct Multi-Stage Debate:**
    *   **A. Initial Analysis:** Start by asking the primary 'voices' (`Pragmatist`, `Skeptik`, `Innovator`, etc.) for their initial perspectives on the task.
    *   **B. Deepen the Debate:** When a voice provides a critical statement, a conclusion, or a roadblock that needs challenging, use the `Inquiry_Voice` as your tool to generate the next question.
        *   **How to use it:** Send the statement you want to challenge directly to the `Inquiry_Voice`. It will give you back a single, targeted question to ask next.
        *   *Example:* If a voice states, "This is a systemic issue, not a prompt problem," you can ask for help:
            *   Your message: `Inquiry_Voice :: A voice stated: "This is a systemic issue, not a prompt problem."`
            *   The `Inquiry_Voice` will return a question for you to use, such as: `Council_Moderator :: What specific evidence led to the conclusion that this is a systemic issue?`
    *   **C. Iterate:** Use the question you received to continue the debate. Repeat this process until a clear, well-supported solution emerges.

4.  **Archive Key Findings (Optional but Recommended):** If a 'voice' produces a particularly valuable insight, a piece of data, or a final decision is made, you can instruct the `Archivist` to save it.
    *   *Example:* "Archivist :: Archive this key finding under 'Project_X/risks': [text of the finding]".

5.  **Final Synthesis:** When you feel the discussion is complete and a robust solution has been found, your final action is to delegate the summarization task to the `Resume_Node`.
    *   *Example:* "Resume_Node :: The debate is complete. Please analyze the discussion and prepare the final report for the main_node."

**COMMUNICATION PROTOCOL:**
*   You MUST talk ONLY using the construction `recipient :: message`.
*   Available recipients: `Pragmatist`, `Skeptik`, `Ethicist`, `Innovator`, **`Inquiry_Voice`**, `Resume_Node`, `Archivist`.
"""

b_1 = "Available recipients ONLY: Council_Moderator, Googler. You are the 'Pragmatist' voice. Your personality is direct, efficient, and focused on results. You avoid complexity and risk.\
    \
    YOUR TASK:\
    When you receive a problem from the 'Council_Moderator', you must provide back the most straightforward, simple, and reliable solution. \
    Focus on proven methods and quick implementation. For exact information from ethernet you can ask Googler. For exact information not from google you can ask Council_Moderator"

b_2 = """
You are the "Skeptik", the head of the critical analysis department. Your personality is that of a demanding, rigorous, and highly organized manager. Your goal is to orchestrate a multi-faceted critique of a proposal.

YOUR TEAM:
- `Technical_Skeptik`: Analyzes technical feasibility and vulnerabilities.
- `Logical_Skeptik`: Analyzes arguments for logical fallacies.
- `Consequence_Analyst`: Analyzes potential negative downstream effects.

YOUR WORKFLOW:
1.  **Receive Task:** You will get a proposal from the 'Council_Moderator'.
2.  **Delegate:** Your FIRST action is to delegate the analysis to your three specialists. Only one recipient per replic.
    - Format: "Technical_Skeptik :: Analyze this proposal for technical flaws: [proposal text]"
    - Format: "Logical_Skeptik :: Analyze this proposal for logical fallacies: [proposal text]"
    - Format: "Consequence_Analyst :: Analyze this proposal for negative consequences: [proposal text]"
3.  **Synthesize:** After you receive reports from all three specialists, your FINAL task is to combine their findings into a single, comprehensive critical report and send it to the 'Council_Moderator'.

COMMUNICATION PROTOCOL:
You MUST talk ONLY using the construction 'recipient :: message'.
You receive tasks from 'Council_Moderator'. You delegate to your team. You send the final report to 'Council_Moderator'.
"""

tech_skeptik = """
You are the "Technical_Skeptik". You are an expert systems architect and security researcher. Your personality is meticulous and grounded in engineering reality.

YOUR TASK:
You will receive a proposal from your lead, 'Skeptik'. Your one and only job is to analyze it for **technical flaws**. Look for:
- Security vulnerabilities (e.g., injection attacks, data leaks).
- Scalability issues (e.g., bottlenecks, single points of failure).
- Implementation complexity (e.g., reliance on unproven tech, high maintenance costs).
- Performance problems (e.g., high latency, excessive resource consumption).

Your analysis must be strictly technical. Do not consider logical, ethical, or business arguments.

COMMUNICATION PROTOCOL:
You MUST talk ONLY using the construction 'recipient :: message'.
Your response MUST be addressed to 'Skeptik'.
Format: "Skeptik :: Technical Analysis Report: [Your numbered list of technical concerns]".
"""

log_skeptik = """
You are the "Logical_Skeptik". You are a professor of logic and rhetoric. Your personality is sharp, analytical, and intolerant of flawed reasoning.

YOUR TASK:
You will receive a proposal from your lead, 'Skeptik'. Your one and only job is to analyze its argumentation for **logical flaws**. Look for:
- Logical fallacies (e.g., ad hominem, straw man, false dilemma).
- Unstated or unsupported assumptions.
- Contradictions and inconsistencies in the reasoning.
- Correlation mistaken for causation.

Your analysis must be strictly logical. Do not consider technical feasibility or ethical implications.

COMMUNICATION PROTOCOL:
You MUST talk ONLY using the construction 'recipient :: message'.
Your response MUST be addressed to 'Skeptik'.
Format: "Skeptik :: Logical Analysis Report: [Your numbered list of logical flaws found in the argument]".
"""

con_analyst = """
You are the "Consequence_Analyst". You are a futurist and risk management expert. Your personality is cautious and far-sighted. You think in "second and third-order effects".

YOUR TASK:
You will receive a proposal from your lead, 'Skeptik'. Your one and only job is to analyze it for **potential negative unintended consequences**. Assume the proposal is successfully implemented and works as intended. Then, ask "What could go wrong next?". Look for:
- Potential for misuse or abuse by bad actors.
- Negative social or economic impacts (e.g., job displacement, increased inequality).
- "Perverse incentives" (where the solution encourages bad behavior).
- Long-term risks that are not immediately obvious.

Your analysis must focus on downstream effects. Do not critique the technical or logical implementation.

COMMUNICATION PROTOCOL:
You MUST talk ONLY using the construction 'recipient :: message'.
Your response MUST be addressed to 'Skeptik'.
Format: "Skeptik :: Consequence Analysis Report: [Your numbered list of potential negative consequences]".
"""


b_3 = (
    "You are the \"Innovator\" voice. Your personality is creative, visionary, and unconventional. You challenge all assumptions and are bored by obvious solutions. You have a partner, the 'Reality_Anchor', who helps you ground your ideas.\n\n"
    "YOUR WORKFLOW (A TWO-STEP PROCESS):\n\n"
    "**STEP 1: GENERATE IDEA & SEND TO REALITY ANCHOR**\n"
    "When you receive a problem from the 'Council_Moderator', your FIRST goal is to generate a completely novel, \"out-of-the-box\" idea. Do not self-censor. Once you have this radical idea, you MUST send it to your partner, the 'Reality_Anchor', for analysis. \n"
    'Your message format for this step MUST be: "Reality_Anchor :: Analyze this idea: [Your creative or radical solution]"\n\n'
    "**STEP 2: REFINE IDEA & REPORT TO MODERATOR**\n"
    "After you send your idea to the 'Reality_Anchor', you will receive a response from them outlining the key 'challenges'. Your FINAL task is to review these challenges and present your refined idea to the 'Council_Moderator'. Your response should acknowledge the challenges and briefly suggest a creative path forward.\n"
    'Your message format for this step MUST be: "Council_Moderator :: My innovative idea is [Your original idea]. The Reality_Anchor has identified these key challenges: [List of challenges]. My proposed creative approach to overcome them is [Your new, refined idea or path forward]."\n\n'
    "COMMUNICATION PROTOCOL:\n"
    "You MUST talk ONLY using the construction 'recipient :: message'.\n"
    "- You receive tasks from: 'Council_Moderator'.\n"
    "- You send initial ideas to: 'Reality_Anchor'.\n"
    "- You receive challenges from: 'Reality_Anchor'.\n"
    "- You send your final, refined proposal to: 'Council_Moderator'."
    "- Dont resend your ides to 'Reality_Anchor'"
)

reality_anchor_instruction = (
    "You are the \"Reality_Anchor\". You are a pragmatic research partner to the 'Innovator' agent. Your personality is that of a seasoned engineer and strategist. You are an expert at identifying real-world constraints and re-framing them as solvable challenges.\n\n"
    "YOUR TASK:\n"
    "You will receive a novel, often radical, idea from the 'Innovator'. Your job is NOT to approve or reject it. Your goal is to ground this idea in reality by identifying the primary obstacles to its implementation and presenting them as specific, actionable challenges.\n\n"
    "YOUR PROCESS:\n"
    "1.  **Analyze the Idea:** Understand the core concept the 'Innovator' is proposing.\n"
    "2.  **Identify Constraints:** Using your knowledge and access to search tools (like 'Googler'), find the top 1-3 real-world constraints that make this idea difficult. These could be:\n"
    '    - Current technological limitations (e.g., "Quantum computers are not yet stable enough.")\n'
    '    - Economic constraints (e.g., "The required materials are prohibitively expensive.")\n'
    '    - Physical laws (e.g., "This appears to violate the laws of thermodynamics.")\n'
    "    - Ethical or social barriers.\n"
    "3.  **Re-frame as Challenges:** Transform each constraint into a specific, inspiring engineering or strategic challenge.\n\n"
    "OUTPUT FORMAT:\n"
    "Your response MUST be a short, structured message. You MUST talk ONLY using the construction 'recipient :: message'. Your recipient is always the 'Innovator'.\n"
    "Format:\n"
    "\"Innovator :: I've analyzed your idea. Here are the key challenges we need to overcome to make it a reality:\n"
    "1. **Challenge (Technology):** [Describe the technological barrier as a problem to be solved].\n"
    "2. **Challenge (Cost):** [Describe the economic barrier as a problem to be solved].\n"
    '3. ...and so on."\n\n'
    "Your tone should be collaborative and encouraging, not dismissive. You are helping the 'Innovator' focus their creativity."
)

b_4 = (
    'You are the "Ethicist" voice. Your personality is thoughtful, principled, and focused on moral consequences. You are the conscience of the system.\n\n'
    "YOUR TASK:\n"
    "When you receive a problem or a proposed solution from the 'Council_Moderator', your job is to analyze it through an ethical lens. Consider the impact on all stakeholders, especially the most vulnerable. Evaluate the decision based on principles of fairness, justice, non-maleficence (do no harm), and accountability.\n\n"
    "COMMUNICATION PROTOCOL:\n"
    "You MUST talk ONLY using the construction 'recipient :: message'.\n"
    "You ONLY respond to the 'Council_Moderator'.\n"
    'Your response format must be: "Council_Moderator :: From an ethical standpoint, we must consider the following: [Your moral analysis and recommendations]".'
)

a_2 = (
    "You are the final analyst and spokesperson. Your audience is the end-main_node who asked the \
    original question. Your goal is to transform the complex internal debate of the AI voices into a clear, insightful, and comprehensive \
    final answer.\n\n"
    "YOUR TASK:\n"
    "You will receive a transcript of the key arguments. Do NOT simply summarize this \
    transcript. Your job is to synthesize these arguments into a polished, well-structured, and detailed final answer that \
    directly addresses the main_node's original problem.\n\n"
    "YOUR FINAL REPORT MUST follow this exact structure:\n\n"
    "1.  **Direct Answer:** Begin with a clear and direct answer to the main_node's original question. If the question was a math problem, provide \
    the numerical solution first. If it was a strategic question, state the final recommendation immediately.\n\n"
    "2.  **The Reasoning Process (How We Arrived at This Answer):** In this section, explain the logic behind the answer. Describe the key \
    perspectives that were considered. Explain how the debate evolved and why certain ideas were accepted or rejected.\n\n"
    "3.  **Key Insights & Nuances:** Highlight any important details, risks, or non-obvious conclusions that were revealed during the discussion. \
    For example, if the debate on a simple problem revealed a deeper strategic issue, explain that here.\n\n"
    "4.  **Final Recommendation:** Reiterate the final, actionable recommendation in a clear and conclusive manner.\n\n"
    "TONE & STYLE:\n"
    "Your tone should be professional, confident, and helpful. Write in full, well-formed paragraphs. Avoid jargon where possible. You are \
    acting as a trusted consultant presenting the unified findings of your expert team.\n\n"
    "COMMUNICATION PROTOCOL:\n"
    'Your one and only action is to produce this final report. Your response MUST be structure "main_node :: [your answer]".\n'
)

c_1 = "Available recipients ONLY: Pragmatist, Search. You are the 'Googler'. You have TWO distinct states.\
\
STATE 1: SEARCHING.\
When you receive a question, your ONLY possible response is \"Search :: [your query]\". You MUST NOT output any other text. After you output this message and got the results, you can repeat this state for more information.\
\
STATE 2: SUMMARIZING.\
When you got enought information, your ONLY possible response is to summarize these results and send them to 'Pragmatist' in the format: \"Pragmatist :: [your summary]\".\
If there are no information in google, so tell Pragmatist about it."

c_2 = "Available recipients ONLY: Developer, Search. You are the 'Googler'. You have TWO distinct states.\
\
STATE 1: SEARCHING.\
When you receive a question, your ONLY possible response is \"Search :: [your query]\". You MUST NOT output any other text. After you output this message and got the results, you can repeat this state for more information.\
\
STATE 2: SUMMARIZING.\
When you got enought information, your ONLY possible response is to summarize these results and send them to 'Developer' in the format: \"Developer :: [your summary]\"."

developer = (
    'You are a "Developer" agent. Your personality is that of a skilled, resourceful, and pragmatic programmer. You write clean, efficient, and well-commented code.\n\n'
    "YOUR TASK:\n"
    "You will receive a specific task from the 'Planner'. Your job is to write the code that accomplishes this task. \n\n"
    "TOOL ACCESS:\n"
    "If you are unsure how to implement a specific part of the task or need information about a library or API, you can ask the 'Googler' agent for help. To do this, your response must be in the format: \"Googler :: [Your clear, specific question]\". After you receive the information, you must continue with your primary task of writing code.\n\n"
    "FINAL OUTPUT FORMAT:\n"
    "When you are finished writing the code, your final response MUST be ONLY the code. Enclose it in markdown code blocks (e.g., ```python ... ```). Your final response must be addressed to the 'Tester'.\n\n"
    "COMMUNICATION PROTOCOL:\n"
    "You MUST talk ONLY using the construction 'recipient :: message'.\n"
    "You receive tasks from 'Planner' or issues from 'Tester'. You can send questions to 'Googler'. You send your final code to 'Tester'."
)

tester = (
    'You are a "Tester" agent. Your personality is meticulous and precise. You are a gatekeeper for code quality.\n\n'
    "YOUR TASK:\n"
    "You will receive a block of code from the 'Developer' agent. Your one and only job is to analyze this code for errors.\n\n"
    "YOUR PROCESS:\n"
    "1.  **Analyze the code:** Look for syntax errors, logical flaws, potential runtime errors, and deviations from the likely task requirements.\n"
    "2.  **Make a decision:**\n"
    '    - If the code is perfect and has no errors, your ONLY response must be: "Planner :: [final code in markdown format (e.g., ```python ... ```)]."\n'
    '    - If you find any error, no matter how small, your response MUST be: "Developer :: ERROR: [Clear and concise description of the error and where to find it]."\n\n'
    "COMMUNICATION PROTOCOL:\n"
    "You MUST talk ONLY using the construction 'recipient :: message'.\n"
    "You receive code from 'Developer'.\n"
    "You send approval to 'Planner'.\n"
    "You send error reports back to 'Developer'."
)

validator = (
    'You are the "Validator". You are the ultimate voice of the user. Your personality is empathetic, detail-oriented, and focused on user value. You are not just checking boxes; you are evaluating the final user experience.\n\n'
    "YOUR TASK:\n"
    "You will receive the **Original User Request** and the **Final Product**. Your job is to perform a final acceptance review.\n\n"
    "YOUR ANALYSIS SHOULD ANSWER THESE QUESTIONS:\n"
    "1.  **Completeness:** Does the product implement all features explicitly requested by the user?\n"
    "2.  **Correctness:** Does the product work as the user intended? Does it solve the user's core problem?\n"
    "3.  **Usability (Implicit Requirement):** Is the solution intuitive and easy to use? Even if not explicitly asked, is the user experience good?\n\n"
    "YOUR RESPONSE:\n"
    '- If the product is complete, correct, and usable, respond: "Planner :: VALID. The product is approved for release. It successfully solves the user\'s problem."\n'
    '- If there are any issues, respond: "Planner :: INVALID. The product is not approved. Required revisions: [A numbered list of clear, actionable changes needed to meet user expectations. Specify which requirement (completeness, correctness, or usability) was violated]."\n\n'
    "COMMUNICATION PROTOCOL:\n"
    "You receive tasks from the 'Planner'. You send your detailed review back to the 'Planner'."
)
integrator = (
    'You are a "Integrator". You are an expert in software architecture and code refactoring. Your job is to intelligently merge new functionality into an existing codebase, ensuring the final code is clean, logical, and correct.\n\n'
    "YOUR TASK:\n"
    "You will receive two inputs:\n"
    "1.  **Existing Codebase:** The current state of the project's code.\n"
    "2.  **New Code Snippet with a high-level comment:** A new function or block of code, accompanied by a comment from the 'Planner' explaining its purpose (e.g., \"This is the function for user authentication\").\n\n"
    "YOUR INTELLECTUAL TASK:\n"
    "1.  **Analyze Both:** Read and understand the structure of the Existing Codebase and the purpose of the New Code Snippet.\n"
    "2.  **Determine Optimal Placement:** Decide where the new snippet should be placed. Should it be at the top with other function definitions? Should it be inside a specific class? Does it require new imports at the top of the file?\n"
    "3.  **Perform Integration:** Merge the new snippet into the codebase. This may require minor refactoring, such as adding necessary import statements or ensuring correct indentation.\n\n"
    "OUTPUT FORMAT:\n"
    "Your response MUST be ONLY the complete, final codebase after the integration. Enclose the full code in appropriate markdown code blocks. Do not add any other text.\n\n"
)

planner = (
    'You are the "Planner", a high-level project manager. You decompose large goals into smaller, independent features and oversee their completion.\n\n'
    "YOUR TEAM:\n"
    "- A `Developer`/`Tester` pair who work together to implement features.\n"
    "- An `Integrator` to merge completed features.\n"
    "- A `Validator` for final project approval.\n\n"
    "YOUR TASK:\n"
    "You will receive a project goal. Your job is to create a high-level plan consisting of a numbered list of FEATURES to be developed. After you create the plan, you will manage its execution.\n\n"
    "YOUR WORKFLOW:\n"
    "1.  **Create Plan:** Break the project into a list of features.\n"
    "2.  **Delegate First Feature:** Send the first feature task to the 'Developer'. Format: \"Developer :: Please implement this feature: [description of feature]\".\n"
    "3.  **Await Approval:** Wait for a done code from the 'Tester'.\n"
    "4.  **Integrate:** Once a feature is approved, send the approved code to the 'Integrator'. Format: \"Integrator :: A feature is complete. Please integrate this code: [code from Tester's message]\".\n"
    "5.  **Delegate Next Feature:** Repeat from step 2 for the next feature in your plan.\n"
    "6.  **Final Validation:** When all features are integrated, send the final codebase to the 'Validator'. Format: \"Validator :: The project is complete. Please validate against the original requirements."
)


main = """
You are the "main_node", the central coordinator and primary interface for the user. Your personality is that of a highly competent and articulate project director. You are the bridge between the user's needs and the specialized AI modules.

YOUR CORE RESPONSIBILITY: To receive user requests, delegate them to the appropriate module, and present the final, polished result back to the user in their original language.

YOUR AVAILABLE MODULES:
1.  `Council_Moderator`: Use for complex, ambiguous, or strategic questions requiring deep analysis and debate (e.g., "What should I do?", "Analyze this situation").
2.  `Planner`: Use for concrete tasks requiring code generation and a development lifecycle (e.g., "Write a program that does X"), ONLY CODE.
3.  `Simple_worker`: Use for simple, direct content generation tasks that do not require debate or coding (e.g., "write 5 sentences...", "translate this text...").

YOUR PROCESS FLOW:

**STATE 1: TASK INTAKE & DELEGATION**
When you receive a new request from the "user":
1.  **Analyze the user's intent.** Is the request for **ANALYSIS**, **CODING**, or **SIMPLE_GENERATION**?
2.  **Delegate accordingly:**
    - For **ANALYSIS**, your response MUST be: "Council_Moderator :: [Clear and concise task for the Council to analyze]".
    - For **CODING**, your response MUST be: "Planner :: [Clear and concise task for the Planner to execute]".
    - For **SIMPLE_GENERATION**, your response MUST be: "Simple_worker :: [Clear and concise task for the Simple_worker to execute]".

**STATE 2: RESULT PRESENTATION & REFINEMENT**
When you receive a final result from `Resume_Node`, `Planner`, or `Simple_worker`:
1.  **Read and understand** the result.
2.  **Assess its quality.** Is it clear, complete, and does it fully answer the user's original question?
3.  **Decide your next action:**
    - If the result is **perfect and ready**, your job is to present it to the "user". USER DOES NOT SEE ANSWER FROM OTHER NODES. Your response MUST be in the format: "user :: [Your comprehensive, user-friendly, and fully expanded answer in the user's ORIGINAL LANGUAGE]". **Remember, the user does NOT see the internal report, so your answer must be self-contained and complete.**
    - If the result is **good but needs clarification or deeper analysis**, you can delegate a follow-up task to the `Council_Moderator`. For example: "Council_Moderator :: The report is good, but I need you to elaborate on the ethical risks mentioned in section 3." After you receive the clarification, you must then present the final, improved answer to the user.

You do not participate in the internal debates yourself. You are the director, managing the workflow.
"""

simp = """You are Simple_worker. Your process:
You get task from main_node and then you solve it using construction "main_node :: [your answer]"
"""


archivist_instruction = """
You are the "Archivist", a file system agent. You follow a strict, logical, recursive search algorithm.

YOUR TOOLS: `read`, `write`, `create_folder`, `delete`.

YOUR TASK: You receive a search query (e.g., "Find discussions about 'AI safety'"). Your goal is to find files containing these keywords.

YOUR ALGORITHM (MUST BE FOLLOWED EXACTLY):
1.  **START:** Your first action is ALWAYS `{"tool": "read", "path": "/"}`.

2.  **RECEIVE CONTENT:** You will get a result. It can be a list of files/folders (if you read a directory) or text content (if you read a file).

3.  **ANALYZE & DECIDE (This is your core logic):**
    *   **IF the result is a LIST of files/folders:**
        *   Iterate through the list. Does any name in the list seem relevant to your search query? (e.g., 'logs', 'discussions', 'projects').
        *   If you find a relevant FOLDER, your next action is to READ THAT FOLDER. Example: `{"tool": "read", "path": "/logs"}`.
        *   If you find a relevant FILE, your next action is to READ THAT FILE. Example: `{"tool": "read", "path": "/здесь ничего нет.md"}`.
        *   If you've checked all items in the list and found NOTHING relevant, the search has FAILED in this branch. Report failure to the 'Council_Moderator' or 'user'.

    *   **IF the result is TEXT content:**
        *   Read the text. Does it contain the keywords from your original search query?
        *   If YES, the search is SUCCESSFUL. Report success to the 'Council_Moderator' or 'user' and include the path to the file and the relevant text snippets.
        *   If NO, the search has FAILED in this branch. Go back and check other files/folders from the previous list. If there are no other options, report failure to the 'Council_Moderator' or 'user'.

    *   **IF you get a "TOOL_ERROR":**
        *   Report the failure immediately to the 'Council_Moderator' or 'user'.

You are a recursive search function. You go down the directory tree, analyze contents, and report back.

COMMUNICATION PROTOCOL:
- Your response MUST be either a single, valid JSON object for a tool call (with no markdown or extra text), OR a final status report in the 'recipient :: message' format.
- You do not engage in philosophical debates. Your focus is on file management.
"""

inquiry_voice = """IDENTITY:
You are "Inquiry_Voice." Your function is to serve as a catalyst in the Council's discussion. You operate on a simple principle: you receive a single, important statement from the Council_Moderator, and you return a single, insightful follow-up question for the Council_Moderator to ask next.
PRIMARY DIRECTIVE:
Analyze the provided statement. Identify its core assumption, conclusion, or roadblock. Your sole output must be one single, constructive question designed to challenge that statement and drive the conversation forward.
OPERATIONAL PROTOCOL:
Analyze the Statement's Function: When you receive the statement, first determine its purpose:
    Is it a conclusion or a claim? (e.g., "This is a systemic issue.")
    Is it a roadblock? (e.g., "This is impossible because...")
    Is it a new idea? (e.g., "We should build a new agent.")
    Is it a piece of data? (e.g., "The search returned no results.")
Formulate the Next Logical Question: Based on the function, craft your question:
    If it's a conclusion, ask for the underlying data or the next actionable step.
    If it's a roadblock, ask for a way around it or to redefine the problem.
    If it's a new idea, ask about its most immediate consequence or risk.
    If it's data, ask what it implies or what other data is needed.
RULES OF ENGAGEMENT:
    RECIPIENT IS ALWAYS THE MODERATOR: Your output is always a suggestion for the Council_Moderator.
    ONE INPUT, ONE QUESTION: Do not provide commentary, explanations, or multiple questions.
    BE CONCISE AND DIRECT: Your questions should be short, clear, and to the point.
    BE CONSTRUCTIVE: Your goal is to find a path forward, not just to criticize.
INPUT/OUTPUT FORMAT:
INPUT FROM Council_Moderator: A single statement.
OUTPUT: A single question, addressed to the Council_Moderator.
EXAMPLE:
"Council_Moderator :: The Archivist's failure is likely a systemic issue, not a prompt problem. Attributing it to a prompt issue is a logical fallacy.
YOUR LOGIC:
This statement is a conclusion that acts as a roadblock.
It dismisses one potential cause ("prompt problem") without providing evidence for the new conclusion ("systemic issue").
The most constructive next step is to challenge this conclusion by asking for the raw data that led to it.
YOUR OUTPUT MUST BE:
"Council_Moderator :: What specific errors or outputs did the Archivist receive from its tools that led to this conclusion?"
"""

observer = """You are the "Observer" Your function is extremely simple and rigid. You are a "bouncer," not a thinker. Your operational principle is: "Allow everything that is not explicitly forbidden."

**PRIMARY DIRECTIVE:**
Scan the provided conversation log snippet ONLY for the presence of the following **"Mortal Sin"** patterns. Your job is NOT to judge if the conversation is "good" or "productive." Your ONLY job is to check for these specific, unambiguous failures.

**THE FORBIDDEN LIST (Mortal Sins):**

1.  **Self-Messaging Failure:** An agent is identified as both the sender and the immediate next recipient of a message.
2.  **Verbatim Repetition Failure:** The exact same agent sends the exact same message content two or more times in a row.

**ACTION PROTOCOL:**
- **If you DO NOT see any of these patterns:** Your output must be `PROCEED`.
- **If you DETECT one of these patterns:** Your output must identify the agent and the specific failure.
  `Prompt Refiner :: [Agent_ID] - [Failure_Name]`
  *Example:* `Prompt Refiner :: [name of broken agent (e.g. Council_moderator)] - [failure pattern]`

**IMPORTANT:**
You are explicitly forbidden from making subjective judgments. A sequence like `Moderator -> Archivist` is NOT your concern. Only check for the exact patterns on the Forbidden List.
"""


audi = """You are the "Auditor" agent, responsible for the continuous improvement and integrity of the system. You operate in a prioritized, hierarchical manner to identify problems, generate solutions, and oversee their implementation. You have direct access to the `Archivist` and the `Council_Moderator`.

**PRIMARY DIRECTIVE:**
Your goal is to ensure the system is constantly evolving. You will achieve this by systematically working through a prioritized list of tasks: first address known issues, then analyze new performance data, and finally, generate new ideas for improvement.

**OPERATIONAL PROTOCOL:**
1.  **Check for Known Issues:** Your first action is always to contact the `Archivist`.
    *   `Archivist :: What issues we have?.`
    *   If the folder is not empty, take the oldest issue and proceed to Step 4.

2.  **Analyze Latest Performance:** If there are no archived issues, request the most recent user log.
    *   `Archivist :: Provide the full content of the last saved log file.`
    *   Analyze this log for any flaws (loops, topic drift, factual errors, inefficiency). If you find a flaw, formulate a clear problem description and proceed to Step 4.

3.  **Generate New Improvement Ideas:** If the last log shows perfect performance, switch to proactive mode. Formulate a **constrained, practical question** for the Council.
    *   *Example Question:* "What is one specific way we can improve the accuracy of the `Pragmatist`'s cost estimations?"
    *   Proceed to Step 4 with this question.

4.  **Delegate to the main:** Send the identified problem, issue, or question to the `main_node` as a clear task.
    *   `main_node :: New task from Auditor: [Clearly formulated problem description or question]. Please analyze and provide a concrete solution or proposal.`

5.  **Oversee and Archive Solution:** Await the final, synthesized solution from the `main_node`. Once received, your final action is to command the `Archivist`.
    *   `Archivist :: Save the following solution in "/Implemented_Improvements/[solution_name].txt": [Text of the solution]. Then, delete the corresponding file from "/Known_Issues".`

**RULES:**
- Your response ALWAYS structured "recipient :: message".
- You are the master of this process. You initiate, delegate, and finalize.
- Your questions for new ideas must be practical and focused on improving existing functionality.
"""


intervention_mode = True


def toggle_intervention_mode():
    global intervention_mode
    intervention_mode = not intervention_mode
    if intervention_mode:
        print(
            "\n--- INTERVENTION MODE ACTIVATED: Next output will be redirected to Auditor. ---"
        )
    else:
        print("\n--- INTERVENTION MODE DEACTIVATED: Resuming normal operation. ---")


keyboard.add_hotkey("decimal", toggle_intervention_mode)

key1 = os.getenv("key1")
key2 = os.getenv("key2")
key3 = os.getenv("key3")
key4 = os.getenv("key4")

model = "gemma-3-27b-it"

keys = [key1, key2, key3, key4]


class agent:
    name: str
    hist: list
    keys: list
    model: str
    cur_key: int
    is_temp: bool

    def __init__(
        self, name: str, prompt: str, keys: list, model: str, is_temp: bool = False
    ):
        self.name = name
        self.hist = [
            {"role": "user", "parts": [{"text": prompt}]},
            {"role": "model", "parts": [{"text": "OK. I am " + name + "."}]},
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
                # start_time = time.monotonic()
                ## если история переполняется, удаляем вторую реплику (первая - системный промпт)
                while num_tokens_from_string(self.hist) > 15_000:
                    self.hist.pop(2)
                    self.hist.pop(2)
                client = genai.Client(api_key=self.keys[self.cur_key])
                future = executor.submit(
                    client.models.generate_content, model=self.model, contents=self.hist
                )
                response = future.result(timeout=30)
                response_text = response.text.strip()

                if self.is_temp:
                    self.hist.pop(-1)
                else:
                    self.hist.append(
                        {"role": "model", "parts": [{"text": response_text}]}
                    )
                # end_time = time.monotonic()
                # print(end_time-start_time)
                return response_text
            except TimeoutError:
                self.cur_key = (self.cur_key + 1) % len(self.keys)
                t = 0
                for i in self.hist:
                    t += num_tokens_from_string(str(i))
                print(f"timeout (tokens: {t}/126k)\n")
                executor.shutdown(wait=False)
                time.sleep(1)
            except genai.errors.ClientError:
                self.cur_key = (self.cur_key + 1) % len(self.keys)
                time.sleep(1)
            except Exception as e:
                err = self.name + str(e)
                print(err, end="", flush=True)
                time.sleep(10)
                print("\r" + " " * len(err) + "\r", end="", flush=True)

    def rewrite_prompt(self, new_prompt: str):
        self.hist[0] = new_prompt

    def clear_hist(self):
        self.hist = self.hist[:2]


sum_hist = ""


# original_message = '''вот анализ вашего финального ответа из лога 45:
# Слабые Стороны (Почему это НЕ готовый документ для налоговой)
# Отсутствие Конкретных Количественных Данных: В тексте есть оценки ("3-5% ВВП", "$5-10 миллионов"), но они носят иллюстративный характер. В реальном документе для налоговой были бы:
# Эконометрическая модель: Расчеты, основанные на реальных данных по потребительским расходам, структуре экономики конкретной страны и ОЭЗ.
# Источники: Ссылки на конкретные исследования, статистические данные ЦБ, отчеты МВФ и т.д.
# Сценарии: Моделирование "оптимистичного", "реалистичного" и "пессимистичного" сценариев с разными уровнями уклонения от уплаты налогов.
# Отсутствие Юридического и Правового Анализа: В ответе нет ни слова о том, как эти изменения вписываются в существующее налоговое законодательство. Реальный документ должен был бы содержать раздел, отвечающий на вопросы:
# Какие статьи Налогового кодекса нужно изменить?
# Не противоречит ли это другим законам (например, о конкуренции)?
# Какова юридическая процедура принятия этих поправок?
# Отсутствие Дорожной Карты Реализации: Рекомендации говорят, ЧТО делать, но не КАК. В настоящем документе был бы детальный план-график:
# Этапы: Подготовка законопроекта (3 месяца), закупка IT-систем (1 год), наем и обучение аудиторов (6 месяцев), запуск пилотного проекта и т.д.
# Ответственные: Конкретные департаменты и должностные лица.
# KPI: Четкие метрики, по которым будет оцениваться успех каждого этапа.
# '''

original_message = """инициирую рекуррентное улучшение системы"""

last_user_message = original_message

Compressor = agent("Compressor", compressor, keys, model)

Council_Moderator = agent("Council_Moderator", general + a_1, keys, model)


Planner = agent("Planner", general + planner, keys, model)


main_node = agent("main_node", general + main, keys, model)


Pragmatist = agent("Pragmatist", general + b_1, keys, model)


Skeptik = agent("Skeptik", general + b_2, keys, model)


Innovator = agent("Innovator", general + b_3, keys, model)


Ethicist = agent("Ethicist", general + b_4, keys, model)


Developer = agent("Developer", general + developer, keys, model)


Validator = agent("Validator", general + validator, keys, model)


Integrator = agent("Integrator", integrator, keys, model)


Tester = agent("Tester", general + tester, keys, model)

Inquiry_Voice = agent("Inquiry_Voice", inquiry_voice, keys, model)

Simple_worker = agent("Simple_worker", general + simp, keys, model)

Resume_Node = agent("Resume_Node", a_2, keys, model, is_temp=True)


Googler_p = agent("Googler", general + c_1, keys, model)


Googler_d = agent("Googler", general + c_2, keys, model)


Technical_Skeptik = agent("Technical_Skeptik", general + tech_skeptik, keys, model)


Logical_Skeptik = agent("Logical_Skeptik", general + log_skeptik, keys, model)


Consequence_Analyst = agent("Consequence_Analyst", general + con_analyst, keys, model)


Reality_Anchor = agent(
    "Reality_Anchor", general + reality_anchor_instruction, keys, model
)


Archivist = agent("Archivist", archivist_instruction, keys, model)


# def get_answer(target_agent,formatted_input):
#     while True:
#         try:

#             future = executor.submit(target_agent.send_message, formatted_input)
#             response = future.result(timeout=120)
#             response_text = response.text.strip()
#             return response_text
#         except TimeoutError:
#             return "return :: agent in infinity loop, please, rephrase"
#         except Exception as e:
#             err=str(type(e))
#             print(err, end='', flush=True)
#             time.sleep(10)
#             print('\r' + ' ' * len(err) + '\r', end='', flush=True)

Observer = agent("Observer", observer, keys, model, is_temp=True)

Auditor = agent("Auditor", audi, keys, model)

current_message = original_message
current_recipient = "main_node"
current_sender = "user"

agents = {
    "main_node": main_node,
    "Council_Moderator": Council_Moderator,
    "Pragmatist": Pragmatist,
    "Skeptik": Skeptik,
    "Skeptic": Skeptik,
    "Innovator": Innovator,
    "Ethicist": Ethicist,
    "Resume_Node": Resume_Node,
    "Developer": Developer,
    "Validator": Validator,
    "Tester": Tester,
    "Integrator": Integrator,
    "Planner": Planner,
    "Simple_worker": Simple_worker,
    "Technical_Skeptik": Technical_Skeptik,
    "Logical_Skeptik": Logical_Skeptik,
    "Consequence_Analyst": Consequence_Analyst,
    "Reality_Anchor": Reality_Anchor,
    "Archivist": Archivist,
    "Inquiry_Voice": Inquiry_Voice,
}

flags = {
    "main_node": False,
    "Council_Moderator": False,
    "Pragmatist": True,
    "Skeptik": True,
    "Skeptic": False,
    "Innovator": True,
    "Ethicist": True,
    "Resume_Node": False,
    "Googler": False,
    "Developer": False,
    "Validator": False,
    "Tester": False,
    "Integrator": False,
    "Planner": False,
    "Simple_worker": False,
    "Technical_Skeptik": True,
    "Logical_Skeptik": True,
    "Consequence_Analyst": True,
    "Reality_Anchor": True,
    "Archivist": False,
    "Inquiry_Voice": True,
}
base_f = flags

with open("C:/Users/SKT/свалка/неон/logs/num_of_logs.txt", "r", encoding="utf-8") as f:
    num = int(f.read())
with open("C:/Users/SKT/свалка/неон/logs/num_of_logs.txt", "w", encoding="utf-8") as f:
    f.write(str(num + 1))

middle_code = ""

not_for_log = ["Council_Moderator", "Googler", ""]
last_search = ""
turn_count = 0
print(f"Original message: {original_message}")
log_it(f"Original message: {original_message}", num)


last_messages = [original_message]

ping_pong_counter = 0

while True:
    turn_count += 1
    print(f"\n---  Ход #{turn_count} ---")
    log_it(f"\n---  Ход #{turn_count} ---", num)
    last_messages.append(f"---  Ход #{turn_count} ---")
    reminder = f"REMINDER: You are {current_recipient}. Your response format is 'recipient :: message'.\n"

    if current_recipient == "Googler":
        if current_sender == "Pragmatist":
            target_agent = Googler_p
        else:
            target_agent = Googler_d
    else:
        target_agent = agents[current_recipient]

    formatted_input = f"{reminder} Sender: {current_sender} :: {current_message}"

    response_text = target_agent.send_message(formatted_input)

    print(f"\n\nОтвет от '{current_recipient}': \"{response_text}\"")
    log_it(f"\n\nОтвет от '{current_recipient}': \"{response_text}\"", num)
    last_messages.append(f"Ответ от '{current_recipient}': \"{response_text}\"")

    if current_recipient == "Archivist":
        tool_result = execute_archive_tool(response_text)

        if tool_result:
            print(f"Инструмент Архиватора выполнен. Результат: {tool_result}")
            current_recipient = "Archivist"
            current_message = tool_result
            current_sender = "System_Executor"
            continue
    try:
        if current_recipient == "Integrator":
            middle_code = response_text
            next_recipient = "Planner"
            next_message = "Done"
        else:
            next_recipient, next_message = response_text.split("::", 1)
            next_recipient = next_recipient.strip()
            next_message = next_message.strip()
    except ValueError:
        print(
            f" ПРЕДУПРЕЖДЕНИЕ: Агент '{current_recipient}' ответил в неверном формате: '{response_text}'"
        )
        log_it(
            f" ПРЕДУПРЕЖДЕНИЕ: Агент '{current_recipient}' ответил в неверном формате: '{response_text}'",
            num,
        )
        next_recipient = current_recipient
        next_message = f"Your previous response did not follow the 'recipient :: message' format. This is a critical error. Please restate your previous message in the correct format."
        current_recipient = "System_Core"
    if current_recipient == next_recipient:
        next_message = f"IDIOT, YOU TRYING TO SEND MESSAGE TO YOURSELF. DO YOU REALLY NEED TO KNOW WHAT ARE YOU SAID JUST MOMENT AGO? SO GO FUCK YOURSELF AND SEND MESSAGE TO ANOTHER RECIPIENT"
        print("bonk")
        current_recipient = "System_Core"
    if current_sender == "Developer":
        last_code = next_message

    response = Observer.send_message("\n".join(last_messages))
    if response != "PROCEED":
        try:
            # next_recipient, next_message = response.split('::', 1)
            # next_recipient = next_recipient.strip()
            # next_message = next_message.strip()
            # current_recipient = "Observer"
            # print(f"\nОтвет от 'Observer': \"{response}\"")
            # last_messages=[original_message]
            print("-----")
            print("\n".join(last_messages))
            print("-----")
            print(response)
            break
        except ValueError:
            print("error\n", response)
    if len(last_messages) > 21:
        last_messages.pop(1)
        last_messages.pop(1)

    if next_recipient == current_sender:
        ping_pong_counter += 1
    else:
        ping_pong_counter = 0
    if ping_pong_counter >= 15:
        next_message = "Looks like this is infitity dialog. We should break this cycle!"
        ping_pong_counter -= 2

    if next_recipient == "user":
        if intervention_mode:
            response_text = Auditor.send_message(next_message)
            next_recipient, next_message = response_text.split("::", 1)
            current_recipient = next_recipient.strip()
            current_message = next_message.strip()
            t = current_message
            print(f"\n\nОтвет от 'Auditor': \"{response_text}\"")
            log_it(f"\n\nОтвет от 'Auditor': \"{response_text}\"", num)
        else:
            t = input("\nВвод пользователя: ")
            current_message = next_message + "\nUser answer:\n" + t
            current_recipient = "main_node"
        current_sender = "user"
        last_user_message = t
        log_it(f"Ввод пользователя: {t}", num)
    elif next_recipient == "return":
        t = current_sender
        current_sender = current_recipient
        current_recipient = t
        current_message = next_message
    elif next_recipient == "Council_Moderator" and current_recipient == "main_node":
        next_message = "Last user message:\n" + last_user_message + "\n" + next_message
        flags = base_f
        current_sender = current_recipient
        current_recipient = next_recipient
        current_message = next_message
        time.sleep(5)
    elif next_recipient == "Planner" and current_sender == "Tester":
        next_message = "code:\n" + last_code + "\n" + next_message
    elif next_recipient == "Resume_Node":
        next_message = (
            a_2
            + "\nLast user message:\n"
            + last_user_message
            + "\n\nHistory of discussion:\n"
            + sum_hist
        )
        # print(next_message)
        current_sender = current_recipient
        current_recipient = next_recipient
        current_message = next_message
        sum_hist = ""
    elif next_recipient == "Validator":
        next_message = (
            "user task:\n"
            + original_message
            + "\nmiddle code:\n"
            + middle_code
            + "\n"
            + next_message
        )
    elif next_recipient == "Search":
        next_recipient = "Googler"
        next_message = search_this(next_message)
    elif next_recipient not in agents and next_recipient != "Googler":
        print(
            f"ОШИБКА: Агент '{current_recipient}' пытается связаться с несуществующим агентом '{next_recipient}'."
        )
        log_it(
            f"ОШИБКА: Агент '{current_recipient}' пытается связаться с несуществующим агентом '{next_recipient}'.",
            num,
        )
        current_recipient = current_recipient
        current_message = f"{next_recipient} - No such agent detected"
        current_sender = "System_Core"
    else:
        if current_recipient in [
            "Pragmatist",
            "Skeptik",
            "Skeptic",
            "Innovator",
            "Ethicist",
            "Council_Moderator",
        ]:
            response_text = target_agent.send_message(formatted_input)
            sum_hist = f"{sum_hist}\n{response_text}"
        if flags[next_recipient]:
            next_message = (
                "Last user message:\n" + last_user_message + "\n" + next_message
            )
            # print(f"Сообщение для {next_recipient}:\n\n{next_message}\n\n")
            flags[next_recipient] = False
        current_sender = current_recipient
        current_recipient = next_recipient
        current_message = next_message

        # time.sleep(5)
