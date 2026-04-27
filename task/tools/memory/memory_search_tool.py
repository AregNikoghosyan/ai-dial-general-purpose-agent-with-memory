import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "search_long_term_memory"

    @property
    def description(self) -> str:
        return (
            "Searches long-term memories about the user using semantic similarity. "
            "ALWAYS call this tool at the beginning of each new conversation to recall relevant context about the user. "
            "Also call it whenever the user's request may be better answered with knowledge of their preferences, "
            "location, habits, or goals (e.g., weather queries → search for user's city; "
            "recommendations → search for preferences). "
            "Use broad queries for general context, specific queries for targeted facts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)

        results: list[MemoryData] = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k
        )

        if not results:
            final_result = "No memories found."
        else:
            lines = ["## Retrieved Memories\n"]
            for mem in results:
                line = f"- **{mem.content}** *(category: {mem.category}"
                if mem.topics:
                    line += f", topics: {', '.join(mem.topics)}"
                line += ")*\n"
                lines.append(line)
            final_result = "".join(lines)

        tool_call_params.stage.append_content(final_result)
        return final_result
