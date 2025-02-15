from django.shortcuts import render
from ninja import NinjaAPI

api = NinjaAPI()


@api.get("/health")
async def health(request):
    return {"status": "ok"}


@api.get("/chat", include_in_schema=False)
def chat(request):
    """Serve the chat interface."""
    return render(request, "api/chat.html")


@api.get("/story", include_in_schema=False)
def story(request):
    """Serve the story generator interface."""
    return render(request, "api/story.html")
