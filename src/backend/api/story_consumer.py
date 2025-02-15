import json
from enum import Enum
from typing import Any
from typing import TypedDict

from channels.generic.websocket import AsyncWebsocketConsumer
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage


class StoryStage(Enum):
    NAME_AGE = "name_age"
    DETAILS = "details"
    THEME = "theme"
    STORY = "story"


class MessageDict(TypedDict):
    type: str
    content: str
    metadata: dict[str, Any]


class StoryConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        """Handle WebSocket connection."""
        self.messages: list[HumanMessage | AIMessage] = []
        self.stage = StoryStage.NAME_AGE
        self.child_name: str | None = None
        self.child_age: int | None = None
        self.child_details: str | None = None
        self.story_theme: str | None = None

        self.llm = ChatAnthropic(
            model_name="claude-3-opus-20240229",
            temperature=0.7,
            timeout=60,
            stop=None,
        )
        await self.accept()

        # Start the conversation by asking for name and age
        await self.send(
            text_data=json.dumps(
                {
                    "type": "story.message",
                    "content": "Hi! I'm excited to create a special bedtime story. To get started, what's the child's name and age?",
                    "metadata": {"role": "assistant"},
                }
            )
        )

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

        # Filter inappropriate content from input
        system_message = (
            "You are a content filter for a children's bedtime story generator. "
            "Your task is to check if the given text contains any inappropriate or adult content. "
            "If it does, explain what's inappropriate. If it's clean, respond with 'SAFE'. "
            "Be especially careful about adult themes, violence, or scary content."
        )
        filter_messages = [
            HumanMessage(content=system_message),
            HumanMessage(content=f"Please check this text: {content}"),
        ]
        filter_response = await self.llm.ainvoke(input=filter_messages)
        if filter_response.content != "SAFE":
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "story.message",
                        "content": "I apologize, but I detected some content that might not be appropriate for a children's story. Could you please rephrase that?",
                        "metadata": {"role": "assistant"},
                    }
                )
            )
            return

        if message_type == "story.message":
            if self.stage == StoryStage.NAME_AGE:
                await self._handle_name_age(content)
            elif self.stage == StoryStage.DETAILS:
                await self._handle_details(content)
            elif self.stage == StoryStage.THEME:
                await self._handle_theme(content)
            elif self.stage == StoryStage.STORY:
                # We don't expect messages in the story stage
                pass

    async def _handle_name_age(self, content: str) -> None:
        """Handle the name and age gathering stage."""
        system_message = (
            "You are helping gather information for a children's bedtime story generator. "
            "Extract the child's name and age from the parent's response. "
            "The age must be between 4-8 years old. "
            "If you can't find both pieces of information or if the age is outside the range, "
            "ask for clarification. "
            "If you find them, respond with 'NAME: <name>, AGE: <age>'"
        )
        messages = [
            HumanMessage(content=system_message),
            HumanMessage(content=content),
        ]
        response = await self.llm.ainvoke(input=messages)

        if isinstance(response.content, str) and response.content.startswith("NAME:"):
            # Successfully extracted name and age
            parts = response.content.split(",")
            self.child_name = parts[0].replace("NAME:", "").strip()
            self.child_age = int(parts[1].replace("AGE:", "").strip())
            self.stage = StoryStage.DETAILS

            # Move to gathering details
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "story.message",
                        "content": f"Great! Now, tell me about {self.child_name}. What are their interests, favorite things, personality traits, or any special details that would make the story more personal?",
                        "metadata": {"role": "assistant"},
                    }
                )
            )
        else:
            # Need clarification
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "story.message",
                        "content": response.content,
                        "metadata": {"role": "assistant"},
                    }
                )
            )

    async def _handle_details(self, content: str) -> None:
        """Handle the details gathering stage."""
        if not self.child_details:
            # First details message
            self.child_details = content
            system_message = (
                f"You are gathering information about {self.child_name}, age {self.child_age}, "
                "for a personalized bedtime story. Based on the parent's response, "
                "ask ONE follow-up question to learn more about an interesting detail "
                "they mentioned. If you have enough details, respond with 'DETAILS_COMPLETE' "
                "and we'll move to theme selection."
            )
        else:
            # Follow-up details message
            self.child_details += f"\n{content}"
            system_message = (
                f"You are gathering information about {self.child_name}, age {self.child_age}, "
                "for a personalized bedtime story. The parent has shared these details:\n\n"
                f"{self.child_details}\n\n"
                "Based on all responses, either ask ONE more follow-up question "
                "or respond with 'DETAILS_COMPLETE' if you have enough information "
                "to create a personalized story."
            )

        messages = [
            HumanMessage(content=system_message),
            HumanMessage(content=content),
        ]
        response = await self.llm.ainvoke(input=messages)

        if isinstance(response.content, str) and response.content == "DETAILS_COMPLETE":
            self.stage = StoryStage.THEME
            # Move to theme selection
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "story.message",
                        "content": "Wonderful! Now, what kind of story would you like? For example, it could be an adventure, fantasy, educational, or any other theme you think would resonate with your child.",
                        "metadata": {"role": "assistant"},
                    }
                )
            )
        else:
            # Ask follow-up question
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "story.message",
                        "content": response.content,
                        "metadata": {"role": "assistant"},
                    }
                )
            )

    async def _handle_theme(self, content: str) -> None:
        """Handle the theme selection stage."""
        self.story_theme = content
        self.stage = StoryStage.STORY

        # Generate the story
        system_message = (
            f"You are creating a bedtime story for {self.child_name}, age {self.child_age}. "
            f"Here are the details about {self.child_name}:\n\n{self.child_details}\n\n"
            f"The requested theme is: {self.story_theme}\n\n"
            "Requirements:\n"
            "- Story should take 5-10 minutes for an adult to read\n"
            "- Use age-appropriate vocabulary\n"
            "- Include educational elements and moral lessons\n"
            "- Have a clear beginning, middle, and end\n"
            "- Incorporate the child's details naturally into the story\n"
            "- Make the child the main character\n"
            "Create an engaging, personalized story that meets these requirements."
        )

        messages = [
            HumanMessage(content=system_message),
        ]
        response = await self.llm.ainvoke(input=messages)

        # Filter the generated story
        filter_messages = [
            HumanMessage(
                content="You are a content filter for children's bedtime stories. "
                "Check if this story contains any inappropriate content, is too scary, "
                "or has any adult themes. If it's clean, respond with 'SAFE'. "
                "If not, explain what needs to be changed."
            ),
            HumanMessage(content=response.content),
        ]
        filter_response = await self.llm.ainvoke(input=filter_messages)

        if (
            isinstance(filter_response.content, str)
            and filter_response.content == "SAFE"
        ):
            # Send the story
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "story.message",
                        "content": response.content,
                        "metadata": {"role": "assistant", "final_story": True},
                    }
                )
            )
        else:
            # Retry with feedback
            retry_message = (
                f"Please revise the story. The content filter found these issues:\n{filter_response.content}\n\n"
                "Generate a new version addressing these concerns."
            )
            messages = [
                HumanMessage(content=system_message),
                HumanMessage(content=retry_message),
            ]
            retry_response = await self.llm.ainvoke(input=messages)

            await self.send(
                text_data=json.dumps(
                    {
                        "type": "story.message",
                        "content": retry_response.content,
                        "metadata": {"role": "assistant", "final_story": True},
                    }
                )
            )
