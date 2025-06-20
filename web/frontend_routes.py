from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from web.templates import get_frontend_html

# Create a router for frontend routes
frontend_router = APIRouter()

@frontend_router.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the frontend HTML at the root path."""
    return HTMLResponse(content=get_frontend_html(), status_code=200)
