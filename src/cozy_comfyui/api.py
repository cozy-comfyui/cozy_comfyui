"""."""

from typing import Any, Dict

from aiohttp import web
from server import PromptServer

from . import \
    logger

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

@PromptServer.instance.routes.get("/cozy_comfyui/message")
async def api_message_post(req) -> Any:
    return web.json_response(ComfyAPIMessage.MESSAGE)

@PromptServer.instance.routes.post(f"/cozy_comfyui/message")
async def api_message_post(req) -> Any:
    json_data = await req.json()
    if (did := json_data.get("id")) is not None:
        data = ComfyAPIMessage.MESSAGE.get(str(did), [])
        data.append(json_data)
        ComfyAPIMessage.MESSAGE[str(did)] = data
    else:
        json_data = {}
    return web.json_response(json_data)

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def parse_reset(ident:str) -> int:
    try:
        data = ComfyAPIMessage.poll(ident, timeout=0.05)
        ret = data.get('cmd', None)
        return ret == 'reset'
    except TimedOutException as e:
        return -1
    except Exception as e:
        logger.error(str(e))
    return 0
