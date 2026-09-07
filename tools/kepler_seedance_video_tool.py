"""
title: Kepler Video Generator (Seedance)
description: Generate short videos from a text prompt, an attached image, or several reference images/audio clips with ByteDance Seedance. Talks to BytePlus ModelArk directly (default) or Atlas Cloud.
author: Kepler Interactive (forked from the Atlas Cloud Media Generator by binyangzhu000-sudo & Haervwe)
author_url: https://github.com/Kepler-Interactive/open-webui-tools
version: 0.4.0
license: MIT
required_open_webui_version: 0.9.1
"""

# Kepler changes versus the upstream atlascloud_media_tool.py:
#   * PROVIDER valve: "byteplus" (ByteDance's own ModelArk API, default) or
#     "atlascloud" (reseller). Same three tool functions either way.
#   * Finished videos are downloaded into Open WebUI file storage (SAVE_TO_OPENWEBUI)
#     because ModelArk result URLs expire after 24 hours.
#   * Video only. Image generation/editing removed so the model has fewer,
#     clearer tools to choose from (KeplerAI already has image generation).
#   * Local filesystem paths are NOT accepted as media inputs any more. The
#     upstream tool would read any path the LLM passed and upload it, which is
#     an exfiltration risk on a shared server.
#   * Open WebUI attachments are resolved through the Files/Storage layer.
#   * "fast" / "best" quality tiers, duration and resolution caps (cost control),
#     and a reference-to-video function that mirrors Dreamina's multi-reference
#     workflow (@Image1 / @Audio1 prompt syntax, up to N references).
#   * Elapsed-time status updates while the render is running.

import asyncio
import base64
import io
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# --- BytePlus ModelArk (ByteDance first-party, international) ---------------
BYTEPLUS_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
BYTEPLUS_TASKS_PATH = "/contents/generations/tasks"
BYTEPLUS_MINI_MODEL = "dreamina-seedance-2-0-mini-260615"
BYTEPLUS_FAST_MODEL = "dreamina-seedance-2-0-fast-260128"
BYTEPLUS_BEST_MODEL = "dreamina-seedance-2-5-260628"

# Observed render throughput on ModelArk (seconds of wall time per second of video), used only
# to show an *estimated* progress bar - the API reports no percentage. Measured 2026-09-07:
# 4s@480p no audio -> 70s, 5s@720p with audio -> 124s (Seedance 2.0 Fast).
EST_SECONDS_PER_VIDEO_SECOND = {"480p": 15.0, "720p": 20.0, "1080p": 38.0}
EST_OVERHEAD_SECONDS = 8.0
EST_MODEL_FACTOR = {"mini": 0.7, "draft": 1.0, "hifi": 1.6}

# Two user-facing tiers ("draft" / "hifi") plus a hidden "mini". Anything else the model passes is
# mapped here; unknown values fall back to the DEFAULT_QUALITY valve.
TIER_ALIASES = {
    "draft": "draft", "fast": "draft", "quick": "draft", "standard": "draft", "default": "draft",
    "hifi": "hifi", "hi-fi": "hifi", "high": "hifi", "high-fidelity": "hifi", "high_fidelity": "hifi",
    "best": "hifi", "final": "hifi", "premium": "hifi",
    "mini": "mini", "cheap": "mini", "cheapest": "mini",
}

log = logging.getLogger("kepler_seedance_video")

# --- Atlas Cloud (reseller) --------------------------------------------------
ATLAS_BASE_URL = "https://api.atlascloud.ai/api/v1"
ATLAS_VIDEO_ENDPOINT = "/model/generateVideo"
ATLAS_UPLOAD_ENDPOINT = "/model/uploadMedia"
ATLAS_PREDICTION_ENDPOINT = "/model/prediction"
ATLAS_FAST_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
ATLAS_BEST_MODEL = "bytedance/seedance-2.5/text-to-video"
ATLAS_IMAGE_TO_VIDEO_MODEL = "bytedance/seedance-2.5/image-to-video"
ATLAS_REFERENCE_MODEL = "bytedance/seedance-2.5/reference-to-video"

COMPLETED_STATUSES = frozenset({"completed", "succeeded", "success"})
FAILED_STATUSES = frozenset({"failed", "error", "cancelled", "canceled", "timeout", "expired"})

VALID_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive")

OWUI_FILE_ID_RE = re.compile(r"/api/v1/files/([0-9a-fA-F-]{8,})")
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

EventEmitter = Optional[Callable[[dict[str, Any]], Awaitable[None]]]


class VideoGenError(RuntimeError):
    """Raised when the provider rejects or fails a generation request."""


class MediaRef:
    """A resolved media input: either raw bytes (+mime) or a public URL."""

    def __init__(self, url: Optional[str] = None, data: Optional[bytes] = None, mime: str = "", name: str = ""):
        self.url = url
        self.data = data
        self.mime = mime
        self.name = name

    def as_data_uri(self) -> str:
        if self.url:
            return self.url
        return f"data:{self.mime};base64,{base64.b64encode(self.data or b'').decode()}"


class Tools:
    class UserValves(BaseModel):
        API_KEY: Optional[str] = Field(
            default=None,
            description="Optional personal API key for the active provider (overrides the shared key).",
            json_schema_extra={"input": {"type": "password"}},
        )

    class Valves(BaseModel):
        PROVIDER: str = Field(
            default="byteplus",
            description="'byteplus' = ByteDance's ModelArk API directly (default). 'atlascloud' = Atlas Cloud reseller.",
        )
        BYTEPLUS_API_KEY: str = Field(
            default="",
            description="BytePlus ModelArk API key (console.byteplus.com > ModelArk > API keys).",
            json_schema_extra={"input": {"type": "password"}},
        )
        BYTEPLUS_BASE_URL: str = Field(default=BYTEPLUS_BASE_URL, description="ModelArk base URL (ap-southeast region).")
        BYTEPLUS_DRAFT_MODEL: str = Field(default=BYTEPLUS_FAST_MODEL, description="ModelArk model for the 'draft' tier.")
        BYTEPLUS_HIFI_MODEL: str = Field(default=BYTEPLUS_BEST_MODEL, description="ModelArk model for the 'hifi' tier.")
        BYTEPLUS_MINI_MODEL: str = Field(default=BYTEPLUS_MINI_MODEL, description="ModelArk model for the hidden 'mini' tier.")
        DEFAULT_QUALITY: str = Field(default="draft", description="Tier used when the caller does not specify one: draft or hifi.")
        PRICE_PER_MILLION_TOKENS_DRAFT: float = Field(
            default=4.30, ge=0, description="USD per 1M video tokens for the draft model, used only to display an estimated cost. 0 hides it."
        )
        PRICE_PER_MILLION_TOKENS_HIFI: float = Field(default=10.70, ge=0, description="USD per 1M video tokens for the hifi model (display only).")
        PRICE_PER_MILLION_TOKENS_MINI: float = Field(default=0.0, ge=0, description="USD per 1M video tokens for the mini model (display only).")
        ATLASCLOUD_API_KEY: str = Field(
            default="",
            description="Atlas Cloud API key (only used when PROVIDER=atlascloud).",
            json_schema_extra={"input": {"type": "password"}},
        )
        ATLAS_BASE_URL: str = Field(default=ATLAS_BASE_URL)
        ATLAS_FAST_MODEL: str = Field(default=ATLAS_FAST_MODEL)
        ATLAS_BEST_MODEL: str = Field(default=ATLAS_BEST_MODEL)
        ATLAS_IMAGE_TO_VIDEO_MODEL: str = Field(default=ATLAS_IMAGE_TO_VIDEO_MODEL)
        ATLAS_REFERENCE_MODEL: str = Field(default=ATLAS_REFERENCE_MODEL)
        DEFAULT_DURATION_SECONDS: int = Field(default=5, ge=4, le=30)
        MAX_DURATION_SECONDS: int = Field(
            default=10, ge=4, le=30, description="Hard cap on clip length to keep costs predictable."
        )
        ALLOWED_RESOLUTIONS: str = Field(
            default="480p,720p,1080p",
            description="Comma-separated resolutions users may request. Anything else falls back to DEFAULT_RESOLUTION.",
        )
        DEFAULT_RESOLUTION: str = Field(default="720p")
        MAX_REFERENCE_IMAGES: int = Field(default=9, ge=1, le=30)
        MAX_REFERENCE_AUDIO: int = Field(default=3, ge=0, le=10)
        POLL_INTERVAL_SECONDS: float = Field(default=5.0, ge=0.5)
        GENERATION_TIMEOUT_SECONDS: float = Field(default=600.0, ge=30.0)
        SAVE_TO_OPENWEBUI: bool = Field(
            default=True,
            description="Download the finished MP4 into Open WebUI file storage (provider links expire after ~24h).",
        )
        RETURN_HTML_EMBED: bool = Field(default=True, description="Render an inline video player in the chat.")

    def __init__(self) -> None:
        self.valves = self.Valves()

    # ------------------------------------------------------------------ config

    def _resolve_config(self, __user__: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        user_key = ""
        if __user__ and isinstance(__user__, dict) and "valves" in __user__:
            v = __user__["valves"]
            try:
                uv = v if isinstance(v, Tools.UserValves) else Tools.UserValves(**(v if isinstance(v, dict) else v.__dict__))
                user_key = (uv.API_KEY or "").strip()
            except Exception:
                user_key = ""

        provider = (self.valves.PROVIDER or "byteplus").strip().lower()
        if provider not in ("byteplus", "atlascloud"):
            provider = "byteplus"
        shared_key = self.valves.BYTEPLUS_API_KEY if provider == "byteplus" else self.valves.ATLASCLOUD_API_KEY
        allowed = {r.strip().lower() for r in self.valves.ALLOWED_RESOLUTIONS.split(",") if r.strip()}
        return {
            "provider": provider,
            "api_key": user_key or shared_key.strip(),
            "byteplus_base_url": self.valves.BYTEPLUS_BASE_URL.rstrip("/"),
            "byteplus_models": {
                "draft": self.valves.BYTEPLUS_DRAFT_MODEL,
                "hifi": self.valves.BYTEPLUS_HIFI_MODEL,
                "mini": self.valves.BYTEPLUS_MINI_MODEL,
            },
            "prices": {
                "draft": self.valves.PRICE_PER_MILLION_TOKENS_DRAFT,
                "hifi": self.valves.PRICE_PER_MILLION_TOKENS_HIFI,
                "mini": self.valves.PRICE_PER_MILLION_TOKENS_MINI,
            },
            "default_quality": TIER_ALIASES.get((self.valves.DEFAULT_QUALITY or "draft").strip().lower(), "draft"),
            "atlas_base_url": self.valves.ATLAS_BASE_URL.rstrip("/"),
            "atlas_fast_model": self.valves.ATLAS_FAST_MODEL,
            "atlas_best_model": self.valves.ATLAS_BEST_MODEL,
            "atlas_image_to_video_model": self.valves.ATLAS_IMAGE_TO_VIDEO_MODEL,
            "atlas_reference_model": self.valves.ATLAS_REFERENCE_MODEL,
            "default_duration": self.valves.DEFAULT_DURATION_SECONDS,
            "max_duration": self.valves.MAX_DURATION_SECONDS,
            "allowed_resolutions": allowed,
            "default_resolution": self.valves.DEFAULT_RESOLUTION,
            "max_ref_images": self.valves.MAX_REFERENCE_IMAGES,
            "max_ref_audio": self.valves.MAX_REFERENCE_AUDIO,
            "poll_interval": self.valves.POLL_INTERVAL_SECONDS,
            "timeout": self.valves.GENERATION_TIMEOUT_SECONDS,
            "save_to_openwebui": self.valves.SAVE_TO_OPENWEBUI,
            "return_html_embed": self.valves.RETURN_HTML_EMBED,
        }

    @staticmethod
    def _clamp_duration(duration: Any, config: Dict[str, Any]) -> int:
        try:
            d = int(duration)
        except (TypeError, ValueError):
            d = config["default_duration"]
        return min(max(d, 4), config["max_duration"])

    @staticmethod
    def _clamp_resolution(resolution: Any, config: Dict[str, Any]) -> str:
        r = str(resolution or "").strip().lower()
        return r if r in config["allowed_resolutions"] else config["default_resolution"]

    @staticmethod
    def _clamp_ratio(ratio: Any) -> str:
        r = str(ratio or "").strip().lower()
        return r if r in VALID_RATIOS else "adaptive"

    # ------------------------------------------------------------------ events

    async def _emit_status(self, emitter: EventEmitter, description: str, *, done: bool) -> None:
        if emitter:
            await emitter({"type": "status", "data": {"description": description, "done": done}})

    async def _emit_progress(
        self, emitter: EventEmitter, label: str, status: str, started: float, config: Dict[str, Any]
    ) -> None:
        """Status line with an ESTIMATED progress bar. The provider reports no percentage, so this is
        elapsed time against a throughput estimate; it parks at 95% if the render runs long."""
        elapsed = time.monotonic() - started
        est = float(config.get("est_total") or 0)
        if est <= 0:
            await self._emit_status(emitter, f"{label}: {status or 'queued'} ({int(elapsed)}s elapsed)", done=False)
            return
        frac = min(elapsed / est, 0.95)
        filled = int(round(frac * 10))
        bar = "▰" * filled + "▱" * (10 - filled)
        tail = "taking longer than usual" if elapsed > est else f"{int(elapsed)}s of ~{int(est)}s"
        await self._emit_status(
            emitter, f"{label}: {bar} ~{int(frac * 100)}% · {tail} (estimate) · {status or 'queued'}", done=False
        )

    # ------------------------------------------------------------------ http

    @staticmethod
    async def _json_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
        try:
            payload = await response.json()
        except (aiohttp.ContentTypeError, ValueError) as exc:
            body = (await response.text()).strip()
            raise VideoGenError(f"Provider returned a non-JSON response ({response.status}): {body[:300]}") from exc
        if not isinstance(payload, dict):
            raise VideoGenError("Provider returned an invalid JSON payload.")
        if response.status >= 400:
            err = payload.get("error")
            detail = (err.get("message") if isinstance(err, dict) else err) or payload.get("message") or payload
            raise VideoGenError(f"Provider request failed ({response.status}): {detail}")
        return payload

    # ------------------------------------------------------------------ media in

    @staticmethod
    async def _read_owui_file(file_id: str) -> Tuple[bytes, str, str]:
        """Read an Open WebUI stored file via the Files/Storage layer (no HTTP)."""
        try:
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage
        except Exception as exc:  # pragma: no cover
            raise VideoGenError(f"Cannot access Open WebUI file storage: {exc}") from exc

        file = await Files.get_file_by_id(file_id)
        if not file or not getattr(file, "path", None):
            raise VideoGenError(f"Attached file {file_id} was not found in Open WebUI storage.")
        local_path = await asyncio.to_thread(Storage.get_file, file.path)
        with open(local_path, "rb") as fh:
            raw = fh.read()
        meta = file.meta or {}
        return raw, (meta.get("name") or file.filename or file_id), (meta.get("content_type") or "application/octet-stream")

    async def _resolve_media(self, media_input: str) -> MediaRef:
        """Turn a data URI, Open WebUI file reference, or public URL into a MediaRef."""
        if not media_input or not isinstance(media_input, str):
            raise VideoGenError("No valid media input provided.")
        s = media_input.strip()

        if s.startswith("data:"):
            header, data_str = s.split(",", 1) if "," in s else ("", s)
            mime = "image/png"
            for candidate in ("image/jpeg", "image/webp", "image/gif", "audio/mpeg", "audio/wav", "video/mp4"):
                if candidate in header:
                    mime = candidate
                    break
            if "audio/mp3" in header:
                mime = "audio/mpeg"
            ext = {"image/jpeg": "jpg", "audio/mpeg": "mp3"}.get(mime, mime.split("/")[-1])
            return MediaRef(data=base64.b64decode(data_str), mime=mime, name=f"input.{ext}")

        m = OWUI_FILE_ID_RE.search(s)
        if m or UUID_RE.match(s):
            raw, name, mime = await self._read_owui_file(m.group(1) if m else s)
            return MediaRef(data=raw, mime=mime, name=name)

        if s.startswith("https://") or s.startswith("http://"):
            return MediaRef(url=s)

        raise VideoGenError("Unsupported media reference. Attach the file to the chat or give a public https:// URL.")

    def _collect_media(
        self,
        messages: Optional[List[Dict[str, Any]]],
        files: Optional[List[Dict[str, Any]]],
        media_type: str,
    ) -> List[str]:
        """Collect image or audio references from __files__ and the latest user message that has any."""
        found: List[str] = []

        def match(f: Dict[str, Any]) -> Optional[str]:
            ctype = ((f.get("file") or {}).get("meta") or {}).get("content_type") or ""
            ftype = f.get("type") or ""
            ok = (media_type == "image" and (ftype == "image" or ctype.startswith("image/"))) or (
                media_type == "audio" and ctype.startswith("audio/")
            )
            return (f.get("url") or f.get("id")) if ok else None

        for f in files or []:
            if isinstance(f, dict):
                u = match(f)
                if u:
                    found.append(u)

        for message in reversed(messages or []):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            hits: List[str] = []
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    b_type = item.get("type")
                    if media_type == "image" and b_type == "image_url":
                        u = item.get("image_url")
                        u = u.get("url") if isinstance(u, dict) else u
                        if u:
                            hits.append(u)
                    elif media_type == "audio" and b_type in ("audio", "input_audio"):
                        a = item.get("input_audio") or item.get("audio") or {}
                        u = (a.get("url") or a.get("data")) if isinstance(a, dict) else None
                        if u:
                            hits.append(u)
            elif isinstance(content, str) and media_type == "image":
                hits.extend(
                    re.findall(r"!\[[^\]]*\]\((https?://[^\s\)]+|data:image/[^\s\)]+|/api/v1/files/[^\s\)]+)\)", content)
                )
            for f in message.get("files") or []:
                if isinstance(f, dict):
                    u = match(f)
                    if u:
                        hits.append(u)
            if hits:
                found.extend(hits)
                break

        seen: set = set()
        out: List[str] = []
        for u in found:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    # ------------------------------------------------------------------ providers

    async def _byteplus_generate(
        self,
        config: Dict[str, Any],
        emitter: EventEmitter,
        label: str,
        model: str,
        prompt: str,
        params: Dict[str, Any],
        first_frame: Optional[MediaRef] = None,
        ref_images: Optional[List[MediaRef]] = None,
        ref_audios: Optional[List[MediaRef]] = None,
    ) -> Dict[str, Any]:
        """Submit a ModelArk video generation task; returns {url, tokens, task_id, seed}."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if first_frame:
            content.append({"type": "image_url", "image_url": {"url": first_frame.as_data_uri()}, "role": "first_frame"})
        for ref in ref_images or []:
            content.append({"type": "image_url", "image_url": {"url": ref.as_data_uri()}, "role": "reference_image"})
        for ref in ref_audios or []:
            content.append({"type": "audio_url", "audio_url": {"url": ref.as_data_uri()}, "role": "reference_audio"})

        payload: Dict[str, Any] = {"model": model, "content": content, "watermark": False, **params}
        # ModelArk only understands 'adaptive' when there is an image to adapt to.
        if payload.get("ratio") == "adaptive" and not (first_frame or ref_images):
            payload["ratio"] = "16:9"

        headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
        base = config["byteplus_base_url"]
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config["timeout"] + 60), headers=headers) as session:
            async with session.post(f"{base}{BYTEPLUS_TASKS_PATH}", json=payload) as response:
                created = await self._json_response(response)
            task_id = created.get("id")
            if not task_id:
                raise VideoGenError(f"ModelArk did not return a task id: {created}")

            started = time.monotonic()
            deadline = started + config["timeout"]
            last_status, last_emit = None, 0.0
            while time.monotonic() < deadline:
                await asyncio.sleep(config["poll_interval"])
                async with session.get(f"{base}{BYTEPLUS_TASKS_PATH}/{task_id}") as response:
                    task = await self._json_response(response)
                status = str(task.get("status", "")).lower()
                if status in COMPLETED_STATUSES:
                    video_url = (task.get("content") or {}).get("video_url")
                    if not video_url:
                        raise VideoGenError("ModelArk task succeeded without a video_url.")
                    usage = task.get("usage") or {}
                    return {
                        "url": video_url,
                        "tokens": int(usage.get("completion_tokens") or usage.get("total_tokens") or 0),
                        "task_id": task_id,
                        "seed": task.get("seed"),
                    }
                if status in FAILED_STATUSES:
                    err = task.get("error") or {}
                    detail = err.get("message") if isinstance(err, dict) else err
                    raise VideoGenError(f"Generation failed: {detail or status}")
                if status != last_status or time.monotonic() - last_emit >= 10:
                    last_status, last_emit = status, time.monotonic()
                    await self._emit_progress(emitter, label, status, started, config)
        raise VideoGenError("Generation timed out. Try a shorter clip or lower resolution.")

    async def _atlas_upload(self, session: aiohttp.ClientSession, base: str, ref: MediaRef) -> str:
        if ref.url:
            return ref.url
        data = aiohttp.FormData()
        data.add_field("file", ref.data, filename=ref.name or "input", content_type=ref.mime or "application/octet-stream")
        async with session.post(f"{base}{ATLAS_UPLOAD_ENDPOINT}", data=data) as resp:
            payload = await self._json_response(resp)
            res = payload.get("data", payload)
            url = res.get("url") or res.get("file_url") if isinstance(res, dict) else None
            if not url:
                raise VideoGenError("Atlas Cloud uploadMedia did not return a URL.")
            return url

    async def _atlas_generate(
        self,
        config: Dict[str, Any],
        emitter: EventEmitter,
        label: str,
        model: str,
        prompt: str,
        params: Dict[str, Any],
        first_frame: Optional[MediaRef] = None,
        ref_images: Optional[List[MediaRef]] = None,
        ref_audios: Optional[List[MediaRef]] = None,
    ) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {config['api_key']}"}
        base = config["atlas_base_url"]
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config["timeout"] + 60), headers=headers) as session:
            payload: Dict[str, Any] = {"model": model, "prompt": prompt, **params}
            if first_frame:
                payload["image_url"] = await self._atlas_upload(session, base, first_frame)
            if ref_images:
                payload["reference_images"] = [await self._atlas_upload(session, base, r) for r in ref_images]
            if ref_audios:
                payload["reference_audios"] = [await self._atlas_upload(session, base, r) for r in ref_audios]

            async with session.post(f"{base}{ATLAS_VIDEO_ENDPOINT}", json=payload) as response:
                created = await self._json_response(response)
            data = created.get("data", created)
            prediction_id = data.get("id") or data.get("request_id") if isinstance(data, dict) else None
            if not prediction_id:
                raise VideoGenError("Atlas Cloud did not return a prediction id.")

            started = time.monotonic()
            deadline = started + config["timeout"]
            last_status, last_emit = None, 0.0
            while time.monotonic() < deadline:
                await asyncio.sleep(config["poll_interval"])
                async with session.get(f"{base}{ATLAS_PREDICTION_ENDPOINT}/{prediction_id}") as response:
                    pred = await self._json_response(response)
                pred = pred.get("data", pred)
                status = str(pred.get("status", "")).lower()
                if status in COMPLETED_STATUSES:
                    outputs = pred.get("outputs") or pred.get("output") or []
                    if isinstance(outputs, str):
                        outputs = [outputs]
                    if not outputs:
                        raise VideoGenError("Atlas Cloud completed without output URLs.")
                    return {"url": outputs[0], "tokens": 0, "task_id": prediction_id, "seed": None}
                if status in FAILED_STATUSES:
                    raise VideoGenError(f"Generation failed: {pred.get('error') or pred.get('message') or status}")
                if status != last_status or time.monotonic() - last_emit >= 10:
                    last_status, last_emit = status, time.monotonic()
                    await self._emit_progress(emitter, label, status, started, config)
        raise VideoGenError("Generation timed out. Try a shorter clip or lower resolution.")

    # ------------------------------------------------------------------ media out

    async def _save_to_openwebui(
        self, video_url: str, user_id: Optional[str], filename: str, extra_meta: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Download the MP4 and register it as an Open WebUI file. Returns the permanent relative URL."""
        if not user_id:
            return None
        try:
            from open_webui.models.files import FileForm, Files
            from open_webui.storage.provider import Storage
        except Exception:
            return None

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            async with session.get(video_url) as resp:
                if resp.status != 200:
                    raise VideoGenError(f"Could not download finished video ({resp.status}).")
                data = await resp.read()
                content_type = resp.headers.get("Content-Type", "video/mp4").split(";")[0] or "video/mp4"

        file_id = str(uuid.uuid4())
        stored_name = f"{file_id}_{filename}"
        _, path = await asyncio.to_thread(Storage.upload_file, io.BytesIO(data), stored_name, {"OpenWebUI-User-Id": user_id, "OpenWebUI-File-Id": file_id})
        await Files.insert_new_file(
            user_id,
            FileForm(
                id=file_id,
                filename=stored_name,
                path=path,
                meta={
                    "name": filename,
                    "content_type": content_type,
                    "size": len(data),
                    "source": "kepler_seedance_video",
                    **(extra_meta or {}),
                },
            ),
        )
        return f"/api/v1/files/{file_id}/content"

    def _render(
        self,
        provider_url: str,
        saved_url: Optional[str],
        config: Dict[str, Any],
        summary: str,
        save_error: Optional[str] = None,
        usage_note: str = "",
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        if saved_url:
            keep_note = f"A permanent copy is saved in KeplerAI at {saved_url} (open it or attach it in later chats)."
        else:
            keep_note = "The provider link expires in about 24 hours, so download the clip if you want to keep it."
            if save_error:
                keep_note += f" (Saving a copy into KeplerAI failed: {save_error}; mention this to the user.)"
        context = (
            f"{summary} {usage_note} Video URL: {provider_url}. {keep_note} "
            "Tell the user the clip is ready in one or two sentences and include the token usage line."
        )
        if config["return_html_embed"]:
            links = f'<a href="{provider_url}" target="_blank" rel="noopener">Download (provider link, ~24h)</a>'
            if saved_url:
                links = f'<a href="{saved_url}" target="_blank" rel="noopener">Download (saved in KeplerAI)</a> &middot; ' + links
            usage_html = f'<span style="opacity:.75"> &middot; {usage_note}</span>' if usage_note else ""
            html = (
                f'<video controls autoplay muted playsinline src="{provider_url}" width="960" '
                f'style="max-width:100%;border-radius:8px"></video>'
                f'<p style="font-family:sans-serif;font-size:12px">{links}{usage_html}</p>'
            )
            return HTMLResponse(content=html, headers={"content-disposition": "inline"}), context
        out = f"{summary} {usage_note}\n\n- [Provider link (expires ~24h)]({provider_url})"
        if saved_url:
            out += f"\n- [Saved in KeplerAI]({saved_url})"
        return out

    # ------------------------------------------------------------------ orchestration

    async def _generate(
        self,
        config: Dict[str, Any],
        emitter: EventEmitter,
        user: Optional[Dict[str, Any]],
        label: str,
        quality: str,
        mode: str,
        prompt: str,
        params: Dict[str, Any],
        first_frame: Optional[MediaRef] = None,
        ref_images: Optional[List[MediaRef]] = None,
        ref_audios: Optional[List[MediaRef]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        if not config["api_key"]:
            which = "BYTEPLUS_API_KEY" if config["provider"] == "byteplus" else "ATLASCLOUD_API_KEY"
            return f"Video generation is not configured yet: an admin must set {which} in the tool's Valves."

        # Quality tier -> model. Multi-reference mode needs Seedance 2.5, so it always runs on 'hifi'.
        tier = TIER_ALIASES.get(str(quality or "").lower().strip(), config["default_quality"])
        if mode == "reference":
            tier = "hifi"
        if config["provider"] == "byteplus":
            # ModelArk uses one model id for text / first-frame / reference inputs.
            model = config["byteplus_models"][tier]
            runner = self._byteplus_generate
        else:
            model = {
                "text": config["atlas_best_model"] if tier == "hifi" else config["atlas_fast_model"],
                "image": config["atlas_image_to_video_model"],
                "reference": config["atlas_reference_model"],
            }[mode]
            runner = self._atlas_generate

        # Wall-time estimate for the progress bar (the API gives no percentage).
        rate = EST_SECONDS_PER_VIDEO_SECOND.get(params["resolution"], 25.0)
        audio_factor = 1.15 if params.get("generate_audio") else 1.0
        config["est_total"] = EST_OVERHEAD_SECONDS + params["duration"] * rate * EST_MODEL_FACTOR[tier] * audio_factor

        await self._emit_status(
            emitter, f"Submitting to {model} ({config['provider']}, ~{int(config['est_total'])}s estimated)", done=False
        )
        try:
            result = await runner(config, emitter, label, model, prompt, params, first_frame, ref_images, ref_audios)
        except (VideoGenError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("kepler_seedance_video: generation failed: %s", exc)
            await self._emit_status(emitter, f"Video generation error: {exc}", done=True)
            return f"Video generation failed: {exc}"

        provider_url = result["url"]
        tokens = int(result.get("tokens") or 0)
        usage_note = ""
        if tokens:
            price = float(config["prices"].get(tier) or 0)
            usage_note = f"{tokens:,} video tokens"
            if price > 0:
                usage_note += f" (≈ ${tokens / 1_000_000 * price:.2f})"
        log.info(
            "kepler_seedance_video: %s tier=%s model=%s duration=%ss res=%s tokens=%s task=%s user=%s",
            config["provider"], tier, model, params["duration"], params["resolution"], tokens,
            result.get("task_id"), (user or {}).get("id"),
        )

        saved_url, save_error = None, None
        if config["save_to_openwebui"]:
            await self._emit_status(emitter, "Saving video to KeplerAI storage", done=False)
            try:
                saved_url = await self._save_to_openwebui(
                    provider_url,
                    (user or {}).get("id"),
                    f"seedance-{time.strftime('%Y%m%d-%H%M%S')}.mp4",
                    {
                        "seedance": {
                            "provider": config["provider"],
                            "model": model,
                            "tier": tier,
                            "task_id": result.get("task_id"),
                            "seed": result.get("seed"),
                            "tokens": tokens,
                            "prompt": prompt,
                            "duration": params["duration"],
                            "resolution": params["resolution"],
                            "ratio": params["ratio"],
                            "provider_url": provider_url,
                            "provider_url_expires_at": int(time.time()) + 24 * 3600,
                        }
                    },
                )
                if not saved_url:
                    save_error = "storage layer unavailable or no user id"
            except Exception as exc:  # saving is best-effort; the provider link still works
                save_error = str(exc)
                log.exception("kepler_seedance_video: saving video into Open WebUI storage failed")

        await self._emit_status(emitter, f"{label}: done", done=True)
        summary = f"Generated a {params['duration']}s {params['resolution']} clip with {model} ({tier} tier) via {config['provider']}."
        return self._render(provider_url, saved_url, config, summary, save_error, usage_note)

    async def _prepare_refs(self, emitter: EventEmitter, refs: List[str], kind: str) -> List[MediaRef]:
        if refs:
            await self._emit_status(emitter, f"Preparing {len(refs)} {kind} reference(s)", done=False)
        return [await self._resolve_media(r) for r in refs]

    # ------------------------------------------------------------------ tools

    async def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "16:9",
        generate_audio: bool = True,
        quality: str = "",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate a short video clip from a text prompt using ByteDance Seedance. Use this whenever the user asks to
        make, create, render, or visualise a video, clip, animation, or motion concept and has NOT attached an image.

        Write the prompt as a shot description: subject, action, setting, camera movement (e.g. slow dolly in,
        handheld, aerial), lighting and style. You can direct timing with timestamps, e.g. "0-2s: ..., 2-5s: ...".
        Put dialogue in quotes if speech is wanted; audio is generated natively.

        :param prompt: Detailed description of the shot to generate.
        :param duration: Clip length in seconds (4-10). Default 5. Longer clips cost proportionally more.
        :param resolution: 480p, 720p or 1080p. Default 720p.
        :param ratio: Aspect ratio: 16:9 (default), 9:16 for social/vertical, 1:1, 4:3, 3:4 or 21:9.
        :param generate_audio: Whether to generate synchronised sound, music and speech. Default true.
        :param quality: Leave empty to use the configured default. "draft" = Seedance 2.0 Fast (quick, cheap iteration). "hifi" = Seedance 2.5 (highest fidelity, several times the cost) - only when the user explicitly asks for high fidelity, high quality or a final version.
        """
        config = self._resolve_config(__user__)
        params = {
            "duration": self._clamp_duration(duration, config),
            "resolution": self._clamp_resolution(resolution, config),
            "ratio": self._clamp_ratio(ratio),
            "generate_audio": bool(generate_audio),
        }
        return await self._generate(config, __event_emitter__, __user__, "Rendering video", quality, "text", prompt, params)

    async def generate_video_from_image(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        duration: int = 5,
        resolution: str = "720p",
        generate_audio: bool = True,
        quality: str = "",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Animate an attached image into a video (image-to-video, the image becomes the first frame). Use this when the
        user has attached ONE image and wants it brought to life or animated. The image is picked up automatically
        from the chat; only pass image_url if the user gave an explicit public https:// URL.

        :param prompt: What should happen in the shot: motion, camera move, mood, any dialogue.
        :param image_url: Optional public https:// URL of the start image. Leave empty to use the attached image.
        :param duration: Clip length in seconds (4-10). Default 5.
        :param resolution: 480p, 720p or 1080p. Default 720p.
        :param generate_audio: Whether to generate synchronised sound. Default true.
        :param quality: Leave empty for the configured default. "draft" (quick, cheap) or "hifi" (Seedance 2.5, highest fidelity, only when explicitly asked).
        """
        config = self._resolve_config(__user__)
        target = image_url or next(iter(self._collect_media(__messages__, __files__, "image")), None)
        if not target:
            return "No image found. Ask the user to attach an image to the message, then call this tool again."
        try:
            first_frame = (await self._prepare_refs(__event_emitter__, [target], "image"))[0]
        except (VideoGenError, OSError) as exc:
            return f"Could not read the attached image: {exc}"
        params = {
            "duration": self._clamp_duration(duration, config),
            "resolution": self._clamp_resolution(resolution, config),
            "ratio": "adaptive",
            "generate_audio": bool(generate_audio),
        }
        return await self._generate(
            config, __event_emitter__, __user__, "Animating image", quality, "image", prompt, params, first_frame=first_frame
        )

    async def generate_video_from_references(
        self,
        prompt: str,
        image_urls: Optional[List[str]] = None,
        audio_urls: Optional[List[str]] = None,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "16:9",
        generate_audio: bool = True,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate a video guided by SEVERAL reference assets (characters, props, environments, style frames, audio),
        like Dreamina's multi-reference mode. Use this when the user attached two or more images, or an audio clip,
        and wants them combined or kept consistent in the video. References are picked up from the chat
        automatically. In the prompt, cite them in attachment order as @Image1, @Image2, @Audio1, e.g.
        "@Image1 walks through the market from @Image2, cinematic, 35mm". Always uses the high-quality model.

        :param prompt: Shot description using @Image1/@Image2/@Audio1 to cite the references.
        :param image_urls: Optional explicit public https:// image URLs. Leave empty to use attachments.
        :param audio_urls: Optional explicit public https:// audio URLs. Leave empty to use attachments.
        :param duration: Clip length in seconds (4-10). Default 5.
        :param resolution: 480p, 720p or 1080p. Default 720p.
        :param ratio: Aspect ratio: 16:9 (default), 9:16, 1:1, 4:3, 3:4, 21:9 or adaptive.
        :param generate_audio: Whether to generate synchronised sound. Default true.
        """
        config = self._resolve_config(__user__)
        images = (list(image_urls or []) or self._collect_media(__messages__, __files__, "image"))[: config["max_ref_images"]]
        audios = (list(audio_urls or []) or self._collect_media(__messages__, __files__, "audio"))[: config["max_ref_audio"]]
        if not images and not audios:
            return "No reference images or audio found. Ask the user to attach them, then call this tool again."
        try:
            ref_images = await self._prepare_refs(__event_emitter__, images, "image")
            ref_audios = await self._prepare_refs(__event_emitter__, audios, "audio")
        except (VideoGenError, OSError) as exc:
            return f"Could not read the reference assets: {exc}"
        params = {
            "duration": self._clamp_duration(duration, config),
            "resolution": self._clamp_resolution(resolution, config),
            "ratio": self._clamp_ratio(ratio),
            "generate_audio": bool(generate_audio),
        }
        return await self._generate(
            config, __event_emitter__, __user__, "Rendering from references", "best", "reference", prompt, params,
            ref_images=ref_images, ref_audios=ref_audios,
        )
