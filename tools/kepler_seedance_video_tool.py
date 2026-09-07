"""
title: Kepler Video Generator (Seedance)
description: Generate short videos from a text prompt, from an attached image, or from several reference images/audio clips using ByteDance Seedance models via the Atlas Cloud API.
author: Kepler Interactive (forked from the Atlas Cloud Media Generator by binyangzhu000-sudo & Haervwe)
author_url: https://github.com/Kepler-Interactive/open-webui-tools
version: 0.1.0
license: MIT
required_open_webui_version: 0.9.1
"""

# Kepler changes versus the upstream atlascloud_media_tool.py:
#   * Video only. Image generation/editing removed so the model has fewer,
#     clearer tools to choose from (KeplerAI already has image generation).
#   * Local filesystem paths are NOT accepted as media inputs any more. The
#     upstream tool would read any path the LLM passed and upload it to Atlas
#     Cloud, which is an exfiltration risk on a shared server.
#   * Open WebUI file attachments are resolved through the Files/Storage
#     layer instead of an unauthenticated HTTP fetch of /api/v1/files/...
#   * "fast" / "best" quality tiers, duration and resolution caps (cost control),
#     and a reference-to-video function that mirrors Dreamina's multi-reference
#     workflow (@Image1 / @Audio1 prompt syntax, up to N references).
#   * Elapsed-time status updates while the render is running.

import asyncio
import base64
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

VIDEO_ENDPOINT = "/model/generateVideo"
UPLOAD_ENDPOINT = "/model/uploadMedia"
PREDICTION_ENDPOINT = "/model/prediction"

DEFAULT_FAST_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
DEFAULT_BEST_MODEL = "bytedance/seedance-2.5/text-to-video"
DEFAULT_IMAGE_TO_VIDEO_MODEL = "bytedance/seedance-2.5/image-to-video"
DEFAULT_REFERENCE_MODEL = "bytedance/seedance-2.5/reference-to-video"

COMPLETED_STATUSES = frozenset({"completed", "succeeded", "success"})
FAILED_STATUSES = frozenset({"failed", "error", "cancelled", "canceled", "timeout"})

VALID_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive")

OWUI_FILE_ID_RE = re.compile(r"/api/v1/files/([0-9a-fA-F-]{8,})")
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

EventEmitter = Optional[Callable[[dict[str, Any]], Awaitable[None]]]


class AtlasCloudError(RuntimeError):
    """Raised when Atlas Cloud rejects or fails a generation request."""


class Tools:
    class UserValves(BaseModel):
        ATLASCLOUD_API_KEY: Optional[str] = Field(
            default=None,
            description="Optional personal Atlas Cloud API key (overrides the shared key).",
            json_schema_extra={"input": {"type": "password"}},
        )

    class Valves(BaseModel):
        ATLASCLOUD_API_KEY: str = Field(
            default="",
            description="Shared Atlas Cloud API key (console.atlascloud.ai).",
            json_schema_extra={"input": {"type": "password"}},
        )
        API_BASE_URL: str = Field(
            default="https://api.atlascloud.ai/api/v1",
            description="Atlas Cloud Media API base URL.",
        )
        FAST_MODEL: str = Field(
            default=DEFAULT_FAST_MODEL,
            description="Text-to-video model for quality='fast' (cheap, ~1-2 min).",
        )
        BEST_MODEL: str = Field(
            default=DEFAULT_BEST_MODEL,
            description="Text-to-video model for quality='best' (Seedance 2.5, ~6x the cost).",
        )
        IMAGE_TO_VIDEO_MODEL: str = Field(
            default=DEFAULT_IMAGE_TO_VIDEO_MODEL,
            description="Model used when an attached image is the first frame.",
        )
        REFERENCE_MODEL: str = Field(
            default=DEFAULT_REFERENCE_MODEL,
            description="Model used for multi-reference (images/audio) generation.",
        )
        DEFAULT_DURATION_SECONDS: int = Field(default=5, ge=4, le=30)
        MAX_DURATION_SECONDS: int = Field(
            default=10,
            ge=4,
            le=30,
            description="Hard cap on clip length to keep costs predictable.",
        )
        ALLOWED_RESOLUTIONS: str = Field(
            default="480p,720p,1080p",
            description="Comma-separated resolutions users may request. Anything else falls back to DEFAULT_RESOLUTION.",
        )
        DEFAULT_RESOLUTION: str = Field(default="720p")
        MAX_REFERENCE_IMAGES: int = Field(default=9, ge=1, le=30)
        MAX_REFERENCE_AUDIO: int = Field(default=3, ge=0, le=10)
        POLL_INTERVAL_SECONDS: float = Field(default=3.0, ge=0.5)
        GENERATION_TIMEOUT_SECONDS: float = Field(default=600.0, ge=30.0)
        RETURN_HTML_EMBED: bool = Field(
            default=True,
            description="Render an inline video player in the chat on completion.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    # ------------------------------------------------------------------ config

    def _resolve_config(self, __user__: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        user_key = ""
        if __user__ and isinstance(__user__, dict) and "valves" in __user__:
            v = __user__["valves"]
            try:
                uv = v if isinstance(v, Tools.UserValves) else Tools.UserValves(**(v if isinstance(v, dict) else v.__dict__))
                user_key = (uv.ATLASCLOUD_API_KEY or "").strip()
            except Exception:
                user_key = ""

        allowed = {r.strip().lower() for r in self.valves.ALLOWED_RESOLUTIONS.split(",") if r.strip()}
        return {
            "api_key": user_key or self.valves.ATLASCLOUD_API_KEY.strip(),
            "base_url": self.valves.API_BASE_URL.rstrip("/"),
            "fast_model": self.valves.FAST_MODEL,
            "best_model": self.valves.BEST_MODEL,
            "image_to_video_model": self.valves.IMAGE_TO_VIDEO_MODEL,
            "reference_model": self.valves.REFERENCE_MODEL,
            "default_duration": self.valves.DEFAULT_DURATION_SECONDS,
            "max_duration": self.valves.MAX_DURATION_SECONDS,
            "allowed_resolutions": allowed,
            "default_resolution": self.valves.DEFAULT_RESOLUTION,
            "max_ref_images": self.valves.MAX_REFERENCE_IMAGES,
            "max_ref_audio": self.valves.MAX_REFERENCE_AUDIO,
            "poll_interval": self.valves.POLL_INTERVAL_SECONDS,
            "timeout": self.valves.GENERATION_TIMEOUT_SECONDS,
            "return_html_embed": self.valves.RETURN_HTML_EMBED,
        }

    @staticmethod
    def _clamp_duration(duration: Any, config: Dict[str, Any]) -> int:
        try:
            d = int(duration)
        except (TypeError, ValueError):
            d = config["default_duration"]
        if d < 4:
            d = 4
        return min(d, config["max_duration"])

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

    # ------------------------------------------------------------------ http

    @staticmethod
    async def _json_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
        try:
            payload = await response.json()
        except (aiohttp.ContentTypeError, ValueError) as exc:
            body = (await response.text()).strip()
            raise AtlasCloudError(
                f"Atlas Cloud returned a non-JSON response ({response.status}): {body[:300]}"
            ) from exc
        if not isinstance(payload, dict):
            raise AtlasCloudError("Atlas Cloud returned an invalid JSON payload.")
        if response.status >= 400:
            detail = payload.get("message") or payload.get("error") or payload
            raise AtlasCloudError(f"Atlas Cloud request failed ({response.status}): {detail}")
        return payload

    @staticmethod
    def _data(payload: dict[str, Any]) -> dict[str, Any]:
        code = payload.get("code")
        if code not in (None, 0, 200):
            detail = payload.get("message") or payload.get("error") or f"code {code}"
            raise AtlasCloudError(f"Atlas Cloud request failed: {detail}")
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise AtlasCloudError("Atlas Cloud returned an invalid response payload.")
        return data

    async def _upload_media(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        data = aiohttp.FormData()
        data.add_field("file", file_bytes, filename=filename, content_type=content_type)
        async with session.post(f"{base_url}{UPLOAD_ENDPOINT}", data=data) as resp:
            res = self._data(await self._json_response(resp))
            url = res.get("url") or res.get("file_url")
            if not url or not isinstance(url, str):
                raise AtlasCloudError("Atlas Cloud uploadMedia did not return a valid URL.")
            return url

    # ------------------------------------------------------------------ media

    @staticmethod
    async def _read_owui_file(file_id: str) -> Tuple[bytes, str, str]:
        """Read an Open WebUI stored file via the Files/Storage layer (no HTTP)."""
        try:
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage
        except Exception as exc:  # pragma: no cover - depends on host app
            raise AtlasCloudError(f"Cannot access Open WebUI file storage: {exc}") from exc

        file = await Files.get_file_by_id(file_id)
        if not file or not getattr(file, "path", None):
            raise AtlasCloudError(f"Attached file {file_id} was not found in Open WebUI storage.")

        local_path = Storage.get_file(file.path)
        with open(local_path, "rb") as fh:
            raw = fh.read()

        meta = file.meta or {}
        content_type = meta.get("content_type") or "application/octet-stream"
        filename = meta.get("name") or file.filename or f"{file_id}"
        return raw, filename, content_type

    async def _ensure_atlas_media_url(
        self, session: aiohttp.ClientSession, base_url: str, media_input: str
    ) -> str:
        """Turn a data URI, Open WebUI file reference, or public URL into an Atlas-hosted URL."""
        if not media_input or not isinstance(media_input, str):
            raise AtlasCloudError("No valid media input provided.")
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
            raw = base64.b64decode(data_str)
            return await self._upload_media(session, base_url, raw, f"input.{ext}", mime)

        m = OWUI_FILE_ID_RE.search(s)
        if m or UUID_RE.match(s):
            file_id = m.group(1) if m else s
            raw, filename, content_type = await self._read_owui_file(file_id)
            return await self._upload_media(session, base_url, raw, filename, content_type)

        if s.startswith("https://") or s.startswith("http://"):
            return s

        raise AtlasCloudError(
            "Unsupported media reference. Attach the file to the chat or give a public https:// URL."
        )

    def _collect_media(
        self,
        messages: Optional[List[Dict[str, Any]]],
        files: Optional[List[Dict[str, Any]]],
        media_type: str,
    ) -> List[str]:
        """Collect image or audio references from the most recent message that has any, plus __files__."""
        found: List[str] = []

        for f in files or []:
            if not isinstance(f, dict):
                continue
            ctype = ((f.get("file") or {}).get("meta") or {}).get("content_type") or ""
            ftype = f.get("type") or ""
            is_image = ftype == "image" or ctype.startswith("image/")
            is_audio = ctype.startswith("audio/")
            if (media_type == "image" and is_image) or (media_type == "audio" and is_audio):
                ref = f.get("url") or f.get("id")
                if ref:
                    found.append(ref)

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
                hits.extend(re.findall(r"!\[[^\]]*\]\((https?://[^\s\)]+|data:image/[^\s\)]+|/api/v1/files/[^\s\)]+)\)", content))
            for f in message.get("files") or []:
                if isinstance(f, dict):
                    ctype = ((f.get("file") or {}).get("meta") or {}).get("content_type") or ""
                    if (media_type == "image" and (f.get("type") == "image" or ctype.startswith("image/"))) or (
                        media_type == "audio" and ctype.startswith("audio/")
                    ):
                        u = f.get("url") or f.get("id")
                        if u:
                            hits.append(u)
            if hits:
                found.extend(hits)
                break

        # de-duplicate, keep order
        seen = set()
        out: List[str] = []
        for u in found:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    # ------------------------------------------------------------------ core

    async def _submit_and_wait(
        self,
        payload: dict[str, Any],
        config: Dict[str, Any],
        emitter: EventEmitter,
        label: str,
    ) -> list[str]:
        if not config["api_key"]:
            raise AtlasCloudError(
                "Atlas Cloud API key is not configured. An admin must set ATLASCLOUD_API_KEY in the tool's Valves."
            )
        headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=config["timeout"] + 30)

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.post(f"{config['base_url']}{VIDEO_ENDPOINT}", json=payload) as response:
                submit = self._data(await self._json_response(response))
            prediction_id = submit.get("id") or submit.get("request_id")
            if not prediction_id:
                raise AtlasCloudError("Atlas Cloud did not return a prediction ID.")

            started = time.monotonic()
            deadline = started + config["timeout"]
            last_emit = 0.0
            last_status: Optional[str] = None
            while time.monotonic() < deadline:
                async with session.get(f"{config['base_url']}{PREDICTION_ENDPOINT}/{prediction_id}") as response:
                    prediction = self._data(await self._json_response(response))
                status = str(prediction.get("status", "")).lower()

                if status in COMPLETED_STATUSES:
                    outputs = prediction.get("outputs") or prediction.get("output") or []
                    if isinstance(outputs, str):
                        outputs = [outputs]
                    if not isinstance(outputs, list) or not all(isinstance(o, str) for o in outputs) or not outputs:
                        raise AtlasCloudError("Atlas Cloud completed without valid output URLs.")
                    return outputs
                if status in FAILED_STATUSES:
                    detail = prediction.get("error") or prediction.get("message") or status
                    raise AtlasCloudError(f"Generation failed: {detail}")

                elapsed = int(time.monotonic() - started)
                if status != last_status or time.monotonic() - last_emit >= 15:
                    last_status = status
                    last_emit = time.monotonic()
                    await self._emit_status(
                        emitter, f"{label}: {status or 'processing'} ({elapsed}s elapsed)", done=False
                    )
                await asyncio.sleep(config["poll_interval"])

        raise AtlasCloudError("Generation timed out. Try a shorter clip or lower resolution.")

    def _render_result(
        self, outputs: List[str], config: Dict[str, Any], summary: str
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        url = outputs[0]
        context = (
            f"{summary} Video URL: {url}\n"
            "Tell the user the clip is ready and remind them the link is hosted by Atlas Cloud, "
            "so they should download it if they want to keep it."
        )
        if config["return_html_embed"]:
            html = (
                f'<video controls autoplay muted playsinline src="{url}" width="960" '
                f'style="max-width:100%;border-radius:8px"></video>'
                f'<p style="font-family:sans-serif;font-size:12px"><a href="{url}" target="_blank" rel="noopener">Download video</a></p>'
            )
            return HTMLResponse(content=html, headers={"content-disposition": "inline"}), context
        return f"{summary}\n\n" + "\n".join(f"- [Download video]({u})" for u in outputs)

    async def _run(
        self,
        payload: Dict[str, Any],
        config: Dict[str, Any],
        emitter: EventEmitter,
        label: str,
        summary: str,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        try:
            outputs = await self._submit_and_wait(payload, config, emitter, label)
        except (AtlasCloudError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(emitter, f"Video generation error: {exc}", done=True)
            return f"Video generation failed: {exc}"
        await self._emit_status(emitter, f"{label}: done", done=True)
        return self._render_result(outputs, config, summary)

    # ------------------------------------------------------------------ tools

    async def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "adaptive",
        generate_audio: bool = True,
        quality: str = "fast",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate a short video clip from a text prompt using ByteDance Seedance. Use this whenever the user asks to
        make, create, render, or visualise a video, clip, animation, or motion concept and has NOT attached an image.

        Write the prompt as a shot description: subject, action, setting, camera movement (e.g. slow dolly in,
        handheld, aerial), lighting and style. You can direct timing with timestamps, e.g. "0-2s: ..., 2-5s: ...".
        Keep dialogue in quotes if speech is wanted; audio is generated natively.

        :param prompt: Detailed description of the shot to generate.
        :param duration: Clip length in seconds (4-10). Default 5. Longer clips cost proportionally more.
        :param resolution: 480p, 720p or 1080p. Default 720p.
        :param ratio: Aspect ratio: 16:9, 9:16, 1:1, 4:3, 3:4, 21:9 or adaptive. Use 9:16 for social/vertical.
        :param generate_audio: Whether to generate synchronised sound, music and speech. Default true.
        :param quality: "fast" (default, cheap, quick iteration) or "best" (Seedance 2.5, higher fidelity, ~6x cost). Only use "best" when the user asks for high quality or a final version.
        """
        config = self._resolve_config(__user__)
        model = config["best_model"] if str(quality).lower().strip() == "best" else config["fast_model"]
        payload = {
            "model": model,
            "prompt": prompt,
            "duration": self._clamp_duration(duration, config),
            "resolution": self._clamp_resolution(resolution, config),
            "ratio": self._clamp_ratio(ratio),
            "generate_audio": bool(generate_audio),
        }
        await self._emit_status(__event_emitter__, f"Submitting to {model}", done=False)
        return await self._run(
            payload, config, __event_emitter__, "Rendering video",
            f"Generated a {payload['duration']}s {payload['resolution']} clip with {model}.",
        )

    async def generate_video_from_image(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "adaptive",
        generate_audio: bool = True,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Animate an attached image into a video (image-to-video). Use this when the user has attached ONE image and
        wants it brought to life, animated, or used as the first frame. The image is picked up automatically from
        the chat; only pass image_url if the user gave an explicit public https:// URL.

        :param prompt: What should happen in the shot: motion, camera move, mood, any dialogue.
        :param image_url: Optional public https:// URL of the start image. Leave empty to use the attached image.
        :param duration: Clip length in seconds (4-10). Default 5.
        :param resolution: 480p, 720p or 1080p. Default 720p.
        :param ratio: Aspect ratio or adaptive (default, follows the image).
        :param generate_audio: Whether to generate synchronised sound. Default true.
        """
        config = self._resolve_config(__user__)
        target = image_url or next(iter(self._collect_media(__messages__, __files__, "image")), None)
        if not target:
            return "No image found. Ask the user to attach an image to the message, then call this tool again."

        await self._emit_status(__event_emitter__, "Uploading reference image", done=False)
        try:
            headers = {"Authorization": f"Bearer {config['api_key']}"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120), headers=headers) as session:
                hosted = await self._ensure_atlas_media_url(session, config["base_url"], target)
        except (AtlasCloudError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(__event_emitter__, f"Image upload error: {exc}", done=True)
            return f"Could not prepare the reference image: {exc}"

        payload = {
            "model": config["image_to_video_model"],
            "prompt": prompt,
            "image_url": hosted,
            "duration": self._clamp_duration(duration, config),
            "resolution": self._clamp_resolution(resolution, config),
            "ratio": self._clamp_ratio(ratio),
            "generate_audio": bool(generate_audio),
        }
        return await self._run(
            payload, config, __event_emitter__, "Animating image",
            f"Generated a {payload['duration']}s {payload['resolution']} clip from the attached image with {payload['model']}.",
        )

    async def generate_video_from_references(
        self,
        prompt: str,
        image_urls: Optional[List[str]] = None,
        audio_urls: Optional[List[str]] = None,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "adaptive",
        generate_audio: bool = True,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate a video guided by SEVERAL reference assets (characters, props, environments, style frames, audio),
        similar to Dreamina's multi-reference mode. Use this when the user attached two or more images, or an audio
        clip, and wants them combined or kept consistent in the video. References are picked up from the chat
        automatically. In the prompt, refer to them in attachment order as @Image1, @Image2, @Audio1, e.g.
        "@Image1 walks through the market from @Image2, cinematic, 35mm".

        :param prompt: Shot description using @Image1/@Image2/@Audio1 to cite the references.
        :param image_urls: Optional explicit public https:// image URLs. Leave empty to use attachments.
        :param audio_urls: Optional explicit public https:// audio URLs. Leave empty to use attachments.
        :param duration: Clip length in seconds (4-10). Default 5.
        :param resolution: 480p, 720p or 1080p. Default 720p.
        :param ratio: Aspect ratio or adaptive.
        :param generate_audio: Whether to generate synchronised sound. Default true.
        """
        config = self._resolve_config(__user__)
        images = list(image_urls or []) or self._collect_media(__messages__, __files__, "image")
        audios = list(audio_urls or []) or self._collect_media(__messages__, __files__, "audio")
        images = images[: config["max_ref_images"]]
        audios = audios[: config["max_ref_audio"]]
        if not images and not audios:
            return "No reference images or audio found. Ask the user to attach them, then call this tool again."

        await self._emit_status(
            __event_emitter__, f"Uploading {len(images)} image and {len(audios)} audio reference(s)", done=False
        )
        try:
            headers = {"Authorization": f"Bearer {config['api_key']}"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300), headers=headers) as session:
                hosted_images = [await self._ensure_atlas_media_url(session, config["base_url"], u) for u in images]
                hosted_audios = [await self._ensure_atlas_media_url(session, config["base_url"], u) for u in audios]
        except (AtlasCloudError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(__event_emitter__, f"Reference upload error: {exc}", done=True)
            return f"Could not prepare the reference assets: {exc}"

        payload: Dict[str, Any] = {
            "model": config["reference_model"],
            "prompt": prompt,
            "duration": self._clamp_duration(duration, config),
            "resolution": self._clamp_resolution(resolution, config),
            "ratio": self._clamp_ratio(ratio),
            "generate_audio": bool(generate_audio),
        }
        if hosted_images:
            payload["reference_images"] = hosted_images
        if hosted_audios:
            payload["reference_audios"] = hosted_audios

        return await self._run(
            payload, config, __event_emitter__, "Rendering from references",
            f"Generated a {payload['duration']}s {payload['resolution']} clip from {len(hosted_images)} image and {len(hosted_audios)} audio reference(s) with {payload['model']}.",
        )
