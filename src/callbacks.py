"""
Purpose: Defines custom callback handlers for logging LLM events and prompts.
"""
import logging
from typing import Any, Dict, Optional
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

logger = logging.getLogger(__name__)


class SimplePromptLogger(BaseCallbackHandler):
    def __init__(self):
        super().__init__(
            event_starts_to_ignore=[], event_ends_to_ignore=[]
        )

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type == CBEventType.LLM and payload:
            log_msg = "\n" + "=" * 20 + " PROMPT SENT TO LLM " + "=" * 20 + "\n"
            if EventPayload.PROMPT in payload:
                log_msg += f"{payload[EventPayload.PROMPT]}\n"
            if EventPayload.MESSAGES in payload:
                for msg in payload[EventPayload.MESSAGES]:
                    log_msg += f"[{msg.role}]: {msg.content}\n"
            log_msg += "=" * 60
            logger.info(log_msg)
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        pass

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass