from inference.self_manager.prompt import (
    SYSTEM_PROMPT, USER_PROMPT, CHILD_SYSTEM_PROMPT,
    CHILD_USER_PROMPT, SUMMARIZE_CONTEXT_PROMPT, FORCED_ANSWER_PROMPT
)
from typing import Any, Dict, List, Callable
from inference.tools import Visit, Search
from transformers import AutoTokenizer
from openai import AsyncOpenAI
import re
import time
import json5
import asyncio
import tiktoken


STATS = {}
STATS["search_main"] = 0
STATS["visit_main"] = 0
STATS["branch_main"] = 0
STATS["sleep_main"] = 0
STATS["kill_main"] = 0
STATS["delete_main"] = 0
STATS["search_child"] = 0
STATS["visit_child"] = 0
STATS["summarize_main"] = 0
STATS["summarize_child"] = 0
STATS["round_main"] = 0
STATS["round_child_avg"] = 0

MSGS = {}
MSGS["main"] = []
MSGS["child"] = {}


LLM_MAX_CONCURRENCY = 20
LLM_SEMAPHORE = asyncio.Semaphore(LLM_MAX_CONCURRENCY)
LLM_HARD_TIMEOUT = 120
_LLM_CLIENT: AsyncOpenAI | None = None


async def _call_server_impl(config, messages):
    global _LLM_CLIENT

    if _LLM_CLIENT is None:
        _LLM_CLIENT = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"],
        )
    
    response = await _LLM_CLIENT.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
    )
    return response.choices[0].message.content


async def call_server(config, messages, max_tries=3):
    for attempt in range(max_tries):
        try:
            async with LLM_SEMAPHORE:
                return await asyncio.wait_for(
                    _call_server_impl(config, messages),
                    timeout=LLM_HARD_TIMEOUT,
                )
        except asyncio.TimeoutError:
            print(f"[call_server] HARD TIMEOUT (>{LLM_HARD_TIMEOUT}s), attempt {attempt + 1}")
        except Exception as e:
            print(f"[call_server] Attempt {attempt + 1} failed: {e}")
        await asyncio.sleep(1 + attempt)
    return "sglang server error!!!"



async def summarize_context(config, question, messages, max_tries=3):
    async with AsyncOpenAI(
        api_key=config["summarize_api_key"],
        base_url=config["summarize_base_url"],
        timeout=config["timeout"],
    ) as client:
        for attempt in range(max_tries):
            try:
                response = await client.chat.completions.create(
                    model=config["summarize_model"],
                    messages=[
                        {"role": "user", "content": SUMMARIZE_CONTEXT_PROMPT.format(
                            question=question,
                            recent_history_messages="\n".join([f"{m['role']}: {m['content']}" for m in messages])
                        )},
                    ],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    max_tokens=config["max_tokens"],
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
                await asyncio.sleep(attempt + 1)
        return "sglang server error!!!"



def count_tokens(messages, model):
    try: 
        tokenizer = AutoTokenizer.from_pretrained(model)
    except Exception as e:
        print(f"[count_tokens] Error: {e}")
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return len(tokenizer.encode(prompt))


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]):
        self._tools[name] = func

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._tools:
            raise KeyError(f"[ToolRegistry] Tool '{name}' not registered")
        return self._tools[name]
    
    def keys(self) -> List[str]:
        return list(self._tools.keys())


TOOLS = ToolRegistry()
TOOLS.register("visit", Visit().call)
TOOLS.register("search", Search().call)


def custom_call_tool(tool_name: str, tool_args: dict, **kwargs):
    tool_args["params"] = tool_args
    return TOOLS.get(tool_name.lower())(tool_args, **kwargs)


def extract_final_answer(content: str) -> str:
    cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    answer_regions = list(re.finditer(r"<answer>(.*?)</answer>", cleaned_content, re.DOTALL))
    if len(answer_regions) == 0:
        return None
    else:
        return answer_regions[-1].group(1).strip()



class ThreadControlBlock:
    def __init__(self,
                 id: str,
                 target: str,
                 allowed_tools: List[str],
                 assigned_context: str,
                 extra_info: str = None,
                 ) -> None:
        self.id = id
        self.target = target
        self.status = "Running" # Running, Success, Failed, Killed
        self.allowed_tools = allowed_tools # tool name list
        self.assigned_context = assigned_context# prefix context assigned by main thread
        self.extra_info = extra_info # extra info assigned by main thread (optional)
        self.start_time = time.time()
        self.result = None
        self._task = None # asyncio.Task
    
    def _runtime(self) -> float:
        return time.time() - self.start_time
    
    def print_info(self) -> str:
        info = ""
        info += f"Thread ID: {self.id}\n"
        info += f"Target: {self.target}\n"
        info += f"Status: {self.status}\n"
        info += f"Allowed Tools: {self.allowed_tools}\n"
        info += f"Assigned Context: {self.assigned_context}\n"
        info += f"Extra Info: {self.extra_info}\n"
        info += f"Runtime: {self._runtime()} Seconds\n"
        info += f"Result: {self.result if self.result else 'Not yet available.'}\n"
        return info



async def child_thread_loop(config, tcb: ThreadControlBlock):
    context: List[str] = []
    context.append({"role": "system", "content": CHILD_SYSTEM_PROMPT})
    context.append({"role": "user", "content": CHILD_USER_PROMPT.format(
        target=tcb.target,
        allowed_tools=",".join(tcb.allowed_tools) if tcb.allowed_tools else "None",
        assigned_context=tcb.assigned_context,
        extra_info=tcb.extra_info if tcb.extra_info else "None"
    )})

    round = 0
    try:
        while round < config["child_max_rounds"]:
            round += 1
            print(f"Child Thread {tcb.id} - Round {round}:")


            context_len = count_tokens(context, config["model"])
            max_gen = config["max_tokens"]
            ctx_limit = config.get("context_window", 32768)
            need_summary = (context_len + max_gen) > ctx_limit

            need_force_answer = (round == config["child_max_rounds"]) # the last round or not

            if need_force_answer:
                context.append({"role": "user", "content": FORCED_ANSWER_PROMPT})

            if not need_force_answer and need_summary:
                STATS["summarize_child"] += 1
                compressed_context = await summarize_context(config, tcb.target, context)
                context = [
                    {"role": "system", "content": CHILD_SYSTEM_PROMPT},
                    {"role": "user", "content": CHILD_USER_PROMPT.format(
                        target=tcb.target,
                        allowed_tools=",".join(tcb.allowed_tools) if tcb.allowed_tools else "None",
                        assigned_context=tcb.assigned_context,
                        extra_info=tcb.extra_info if tcb.extra_info else "None"
                    )},
                    {"role": "user", "content": f"The following is a compressed summary of the previous conversation.\n\n{compressed_context}"}
                ]


            content = await call_server(config, context) # Think & Act
            if content == "sglang server error!!!":
                tcb.status = "Failed"
                tcb.result = "Error: The child thread is failed. The sglang server is error."
                STATS["round_child_avg"] += round
                MSGS["child"][tcb.id] = context
                return
            
            context.append({"role": "assistant", "content": content})
            print(f"[DEBUG][Child Thread {tcb.id}] Act:\n {content}")

            if "<answer>" in content and "</answer>" in content: # may contain the final answer
                answer = extract_final_answer(content)
                if answer is not None:
                    tcb.result = answer
                    tcb.status = "Success"
                    STATS["round_child_avg"] += round
                    MSGS["child"][tcb.id] = context
                    return
                # otherwise, it means the final answer is not found -> may not be the last round -> continue

            if "<tool_call>" in content and "</tool_call>" in content:
                try:
                    tool_info_list = re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.S)
                    for tool_info in tool_info_list:
                        tool_call = json5.loads(tool_info)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})

                        if tool_name.lower() in TOOLS.keys():
                            if tool_name.lower() == "search":
                                STATS["search_child"] += 1
                            elif tool_name.lower() == "visit":
                                STATS["visit_child"] += 1
                            result = custom_call_tool(tool_name, tool_args)

                        else:
                            raise ValueError(f"Tool '{tool_name}' not registered")

                except Exception as e:
                    result = f"Error: {e}. The tool call is failed. Please try again or try to use other tools."

            # Observe
            context.append({
                "role": "user",
                "content": f"<tool_response>\n{result}\n</tool_response>"
            })
        
        # ran out of rounds
        tcb.status = "Failed"
        tcb.result = f"Error: The child thread is failed. The maximum number of rounds is reached."
        STATS["round_child_avg"] += round
        MSGS["child"][tcb.id] = context
    
    except asyncio.CancelledError:
        tcb.status = "Killed"
        tcb.result = "Killed by main"
        STATS["round_child_avg"] += round
        MSGS["child"][tcb.id] = context
    
    except Exception as e:
        tcb.status = "Failed"
        tcb.result = f"Error: {e}. The child thread is failed."
        STATS["round_child_avg"] += round
        MSGS["child"][tcb.id] = context




class MainThreadAgent:
    def __init__(self, config) -> None:
        self.config = config
        self.tcbs: Dict[str, ThreadControlBlock] = {} # thread id : TCB
        self.context: List[Dict[str, Any]] = []
        self._lock: asyncio.Lock = asyncio.Lock()
    
    def _get_tcb_list(self) -> str:
        prompt = ""
        for tcb in self.tcbs.values():
            prompt += tcb.print_info()
            prompt += "\n\n\n"
        return prompt
    
    async def run(self, task: str):
        self.context.append({"role": "system", "content": SYSTEM_PROMPT})
        self.context.append({"role": "user", "content": USER_PROMPT.format(task=task)})

        round = 0
        while round < self.config["max_rounds"]:
            round += 1
            print(f"Main Thread - Round {round}:")


            context_len = count_tokens(self.context, self.config["model"])
            max_gen = self.config["max_tokens"]
            ctx_limit = self.config.get("context_window", 32768)
            need_summary = (context_len + max_gen) > ctx_limit

            need_force_answer = (round == self.config["child_max_rounds"]) # the last round or not

            if need_force_answer:
                self.context.append({"role": "user", "content": FORCED_ANSWER_PROMPT})

            if not need_force_answer and need_summary:
                STATS["summarize_main"] += 1
                compressed_context = await summarize_context(self.config, task, self.context)
                self.context = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(task=task)},
                    {"role": "user", "content": f"The following is a compressed summary of the previous conversation.\n\n{compressed_context}"}
                ]


            # Think & Act
            content = await call_server(self.config, self.context)
            self.context.append({"role": "assistant", "content": content})
            print(f"[DEBUG][Main Thread] Act:\n {content}")

            if "<answer>" in content and "</answer>" in content: # may contain the final answer
                answer = extract_final_answer(content)
                if answer is not None:
                    break
                # otherwise, it means the final answer is not found -> may not be the last round -> continue

            if "<tool_call>" in content and "</tool_call>" in content:
                try:
                    tool_info_list = re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.S)
                    for tool_info in tool_info_list:
                        tool_call = json5.loads(tool_info)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})

                        result = "None."
                        if tool_name.lower() in TOOLS.keys():
                            if tool_name.lower() == "search":
                                STATS["search_main"] += 1
                            elif tool_name.lower() == "visit":
                                STATS["visit_main"] += 1
                            result = custom_call_tool(tool_name, tool_args)

                        elif tool_name.lower() == "branch":
                            STATS["branch_main"] += 1
                            tcb = ThreadControlBlock(**tool_args)
                            async with self._lock:
                                if len([t for t in self.tcbs.values() if t.status.lower() == "running"]) >= self.config["max_concurrent_branches"]:
                                    result = f"branch rejected: the maximum number of concurrent branches is reached. (current: {len([t for t in self.tcbs.values() if t.status.lower() == 'running'])}/{self.config['max_concurrent_branches']})"
                                else:
                                    # create and schedule child task
                                    self.tcbs[tcb.id] = tcb
                                    loop = asyncio.get_event_loop()
                                    tcb._task = loop.create_task(child_thread_loop(self.config, tcb))
                                    result = f"branch {tcb.id} started"

                        elif tool_name.lower() == "sleep":
                            STATS["sleep_main"] += 1
                            dur = float(tool_args.get("sleep_duration", 2))
                            await asyncio.sleep(dur)
                            result = f"slept for {dur}s"

                        elif tool_name.lower() == "kill":
                            STATS["kill_main"] += 1
                            target_id = tool_args.get("id", "")
                            async with self._lock:
                                tcb = self.tcbs.get(target_id)
                                if not tcb:
                                    result = f"branch {target_id} not found"
                                else:
                                    if tcb._task and not tcb._task.done():
                                        tcb._task.cancel()
                                        tcb.status = "Killed"
                                        tcb.result = "Killed by main"
                                        result = f"branch {target_id} cancelled"
                                    else:
                                        result = f"branch {target_id} has been marked {tcb.status}"

                        elif tool_name.lower() == "delete":
                            STATS["delete_main"] += 1
                            target_id = tool_args.get("id", "")
                            tcb = self.tcbs.get(target_id)
                            if not tcb:
                                result = f"branch {target_id} not found"
                            else:
                                if tcb._task and not tcb._task.done():
                                    tcb._task.cancel()
                                tcb.status = "Deleted"
                                result = f"branch {target_id} deleted"

                        else:
                            raise ValueError(f"Tool '{tool_name}' not registered")

                except Exception as e:
                    result = f"Error: {e}. The tool call is failed. Please try again or try to use other tools."

            # Observe
            # Remove the TCB list in the previous round to prevent the context from growing large
            if round > 1 and "<tcb_list>" in self.context[-2]["content"]:
                self.context[-2]["content"] = self.context[-2]["content"].split("<tcb_list>")[0]
            
            tcb_list = self._get_tcb_list()
            if tcb_list == "":
                self.context.append({
                    "role": "user",
                    "content": f"<tool_response>\n{result}\n</tool_response>\n\n<tcb_list>\nYou have no sub-threads created. Please create the sub-threads as soon as possible to support more comprehensive research.\n</tcb_list>"
                })
            else:
                self.context.append({
                    "role": "user",
                    "content": f"<tool_response>\n{result}\n</tool_response>\n\n<tcb_list>\n{self._get_tcb_list()}\n</tcb_list>"
                })
            print(f"[DEBUG][Main Thread] Observe:\n<tcb_list>\n{self._get_tcb_list()}\n</tcb_list>")


        async with self._lock:
            for tcb in self.tcbs.values():
                if tcb._task and not tcb._task.done():
                    tcb._task.cancel()
                    tcb.status = "Killed"
                    tcb.result = "Killed by main"
        STATS["round_main"] += round
        STATS["round_child_avg"] = STATS["round_child_avg"] / STATS["branch_main"] if STATS["branch_main"] > 0 else 0
        MSGS["main"] = self.context
        if "<answer>" in self.context[-1]["content"] and "</answer>" in self.context[-1]["content"]:
            answer = extract_final_answer(self.context[-1]["content"])
            if answer is None:
                return "No answer found."
            else:
                return answer
        else:
            return "No answer found."



async def run_one(question, config):
    global STATS, MSGS

    STATS = {}
    STATS["search_main"] = 0
    STATS["visit_main"] = 0
    STATS["branch_main"] = 0
    STATS["sleep_main"] = 0
    STATS["kill_main"] = 0
    STATS["delete_main"] = 0
    STATS["search_child"] = 0
    STATS["visit_child"] = 0
    STATS["summarize_main"] = 0
    STATS["summarize_child"] = 0
    STATS["round_main"] = 0
    STATS["round_child_avg"] = 0

    MSGS = {}
    MSGS["main"] = []
    MSGS["child"] = {}

    agent = MainThreadAgent(config)
    answer = await agent.run(question)
    return answer, STATS, MSGS


