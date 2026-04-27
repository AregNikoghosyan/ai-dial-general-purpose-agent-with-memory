from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class DeleteMemoryTool(BaseTool):
    """
    Tool for deleting all long-term memories about the user.

    This permanently removes all stored memories from the system.
    Use with caution - this action cannot be undone.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "delete_all_long_term_memories"

    @property
    def description(self) -> str:
        return (
            "Permanently deletes ALL long-term memories stored about the user. "
            "Use this tool ONLY when the user explicitly asks to delete, clear, wipe, or forget all memories. "
            "This action is irreversible — all stored facts about the user will be permanently removed. "
            "After deletion, the agent will have no recollection of any previous conversations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {}
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        result = await self.memory_store.delete_all_memories(
            api_key=tool_call_params.api_key
        )
        tool_call_params.stage.append_content(result)
        return result
