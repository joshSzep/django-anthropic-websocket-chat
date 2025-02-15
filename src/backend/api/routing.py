from django.urls import re_path

from backend.api.consumers import ChatConsumer
from backend.api.story_consumer import StoryConsumer

websocket_urlpatterns = [
    re_path(r"ws/chat/$", ChatConsumer.as_asgi()),
    re_path(r"ws/story/$", StoryConsumer.as_asgi()),
]
