"""."""

from typing import Any, Dict
from aiohttp import web
from server import PromptServer

from .. import PACKAGE

# ==============================================================================
# === SERVER ===
# ==============================================================================

class TimedOutException(Exception): pass

class ComfyAPIMessage:
    # Messages are keyed on Node id#: List[Any]
    MESSAGE = {}

    @classmethod
    def poll(cls, ident: str) -> Any:
        """This is used on node execute runs to check if there are any stored messages"""
        return cls.MESSAGE.pop(ident, [])

def comfy_api_post(route:str, ident:str, data:Dict[str, Any]) -> None:
    data['id'] = ident
    PromptServer.instance.send_sync(route, data)

@PromptServer.instance.routes.post(f"/{PACKAGE.lower()}")
async def api_message_post(req) -> Any:
    json_data = await req.json()
    if (did := json_data.get("id")) is not None:
        data = ComfyAPIMessage.MESSAGE.get(str(did), [])
        data.append(json_data)
        ComfyAPIMessage.MESSAGE[str(did)] = data
        return web.json_response(json_data)
    return web.json_response({})
