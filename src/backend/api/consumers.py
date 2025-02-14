import json
from typing import Any
from typing import TypedDict

from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage


class MessageDict(TypedDict):
    type: str
    content: str
    metadata: dict[str, Any]


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        """Handle WebSocket connection."""
        self.chat_history: list[HumanMessage | AIMessage] = []
        self.message_history: BaseChatMessageHistory = InMemoryChatMessageHistory()
        self.llm = ChatAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            model_name="claude-3-opus-20240229",
            temperature=0.7,
            timeout=60,
            stop=None,
        )
        await self.accept()

    async def disconnect(self, code: int | None = None) -> None:
        """Handle WebSocket disconnection."""
        pass

    async def receive(
        self,
        text_data: str | None = None,
        bytes_data: bytes | None = None,
    ) -> None:
        """Handle incoming WebSocket messages."""
        if not text_data:
            return

        data = json.loads(text_data)
        message_type = data.get("type")
        content = data.get("content", "")

        if message_type == "chat.message":
            # Send thinking indicator
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "thinking.start",
                    }
                )
            )

            # Add user message to history
            human_message = HumanMessage(content=content)
            self.chat_history.append(human_message)
            self.message_history.add_message(human_message)

            # Get AI response
            response = await self.llm.apredict_messages(
                messages=self.chat_history,  # type: ignore
            )
            self.chat_history.append(response)  # type: ignore
            self.message_history.add_message(response)

            # Check if we need to summarize
            if self._should_summarize():
                await self._summarize_history()

            # Send AI response
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "chat.message",
                        "content": response.content,
                        "metadata": {
                            "role": "assistant",
                            "can_rewind": True,
                        },
                    }
                )
            )

        elif message_type == "chat.rewind":
            # Get message index to rewind to
            rewind_index = int(data.get("index", 0))
            if 0 <= rewind_index < len(self.chat_history):
                # Truncate history to that point
                self.chat_history = self.chat_history[: rewind_index + 1]
                self.message_history = InMemoryChatMessageHistory()
                for message in self.chat_history:
                    self.message_history.add_message(message)
                await self.send(
                    text_data=json.dumps(
                        {
                            "type": "chat.rewind",
                            "index": rewind_index,
                        }
                    )
                )

    async def _summarize_history(self) -> None:
        """Summarize chat history when it gets too long."""
        system_prompt = """Please summarize the conversation history above in a concise way that preserves the key information and context needed to continue the conversation. Focus on the most important points and decisions made."""

        # Get summary from Claude
        summary_msg = HumanMessage(
            content=f"{system_prompt}\n\nConversation to summarize:\n"
            + "\n".join(str(msg.content) for msg in self.chat_history[:-2])
        )
        summary = await self.llm.apredict_messages(messages=[summary_msg])

        # Replace old history with summary
        self.chat_history = [
            AIMessage(content=f"Previous conversation summary: {summary.content}"),
            *self.chat_history[-2:],
        ]
        self.message_history = InMemoryChatMessageHistory()
        for message in self.chat_history:
            self.message_history.add_message(message)

        # Notify client about summarization
        await self.send(
            text_data=json.dumps(
                {
                    "type": "chat.summarized",
                    "content": summary.content,
                }
            )
        )

    def _should_summarize(self) -> bool:
        """Check if we should summarize the conversation."""
        # Rough estimate: average of 4 chars per token
        total_chars = sum(len(msg.content) for msg in self.chat_history)
        estimated_tokens = total_chars / 4
        # Claude-3-opus-20240229 has a 200k context window
        # We'll summarize at 90% = 180k tokens
        # return estimated_tokens > 180000
        return estimated_tokens > 1000
