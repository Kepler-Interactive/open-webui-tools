"""
title: Kepler Video Generator (Seedance)
description: Generate, extend and edit short videos with ByteDance Seedance from text, attached images, reference images/audio/video. Talks to BytePlus ModelArk directly (default) or Atlas Cloud.
author: Kepler Interactive (forked from the Atlas Cloud Media Generator by binyangzhu000-sudo & Haervwe)
author_url: https://github.com/Kepler-Interactive/open-webui-tools
version: 0.11.2
license: MIT
required_open_webui_version: 0.9.1
"""

# Kepler changes versus the upstream atlascloud_media_tool.py:
#   * PROVIDER valve: "byteplus" (ByteDance's own ModelArk API, default) or
#     "atlascloud" (reseller). Same tool functions either way.
#   * Finished videos are downloaded into Open WebUI file storage (SAVE_TO_OPENWEBUI)
#     because ModelArk result URLs expire after 24 hours. Rich metadata is kept on the
#     file (task id, seed, prompt, provider URL + expiry, chat id) so later edit / extend
#     calls can find "the last clip" and hand ModelArk a URL it can fetch.
#   * extend_video / edit_video / reference video (Seedance "Video 1" convention).
#     ModelArk only accepts reference videos as public web URLs (base64 is rejected),
#     so sources must be clips generated here within 24h or public https links.
#   * Video only. Image generation/editing removed (KeplerAI already has image generation).
#   * Local filesystem paths are NOT accepted as media inputs (exfiltration risk).
#   * Open WebUI attachments are resolved through the Files/Storage layer.
#   * Two quality tiers ("draft" / "hifi", hidden "mini"), duration and resolution caps,
#     estimated progress bar, token usage + estimated cost in the result.

import asyncio
import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
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
VIDEO1_RE = re.compile(r"video\s*1\b", re.IGNORECASE)

# Observed render throughput on ModelArk (seconds of wall time per second of video), used only
# to show an *estimated* progress bar - the API reports no percentage. Measured 2026-09-07:
# 4s@480p no audio -> 70s, 5s@720p with audio -> 124s, 5s@480p with a reference video -> 86-124s.
EST_SECONDS_PER_VIDEO_SECOND = {"480p": 15.0, "720p": 20.0, "1080p": 38.0}
EST_OVERHEAD_SECONDS = 8.0
EST_MODEL_FACTOR = {"mini": 0.7, "draft": 1.0, "hifi": 1.6}
EST_VIDEO_INPUT_FACTOR = 1.4

PROVIDER_URL_TTL_SECONDS = 24 * 3600

# Two user-facing tiers ("draft" / "hifi") plus a hidden "mini". Anything else the model passes is
# mapped here; unknown values fall back to the DEFAULT_QUALITY valve.
TIER_ALIASES = {
    "draft": "draft", "fast": "draft", "quick": "draft", "standard": "draft", "default": "draft",
    "hifi": "hifi", "hi-fi": "hifi", "high": "hifi", "high-fidelity": "hifi", "high_fidelity": "hifi",
    "high fidelity": "hifi", "high quality": "hifi", "high-quality": "hifi", "hq": "hifi",
    "best": "hifi", "final": "hifi", "premium": "hifi",
    "mini": "mini", "cheap": "mini", "cheapest": "mini",
}

log = logging.getLogger("kepler_seedance_video")

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
            default=30, ge=4, le=30,
            description="Hard cap on clip length (cost control). Seedance 2.0 models are further limited to 15s by the API; 2.5 allows 30s.",
        )
        ALLOWED_RESOLUTIONS: str = Field(
            default="480p,720p,1080p",
            description="Comma-separated resolutions users may request. Anything else falls back to DEFAULT_RESOLUTION.",
        )
        DEFAULT_RESOLUTION: str = Field(default="720p")
        MAX_REFERENCE_IMAGES: int = Field(default=9, ge=1, le=30)
        MAX_REFERENCE_VIDEOS: int = Field(default=3, ge=0, le=10)
        MAX_REFERENCE_AUDIO: int = Field(default=3, ge=0, le=10)
        POLL_INTERVAL_SECONDS: float = Field(default=5.0, ge=0.5)
        GENERATION_TIMEOUT_SECONDS: float = Field(default=600.0, ge=30.0)
        SAVE_TO_OPENWEBUI: bool = Field(
            default=True,
            description="Download the finished MP4 into Open WebUI file storage (provider links expire after ~24h).",
        )
        JOIN_EXTENSIONS: bool = Field(
            default=True,
            description="After extend_video, stitch source + continuation into one clip with ffmpeg (needs ffmpeg in the container).",
        )
        JOIN_CUT_SEARCH_SECONDS: float = Field(
            default=1.5, ge=0.0, le=4.0,
            description="Search this many seconds at the start of a continuation for the frame that best matches the source's last frame, skip the near-static onset, and cut there. 0 disables.",
        )
        JOIN_CROSSFADE_SECONDS: float = Field(
            default=0.125, ge=0.0, le=2.0,
            description="Blend length at the cut point. With the anchor and geometry match in place a very short blend (3 frames) looks cleanest; longer blends read as a brief defocus. 0 = hard cut. Values under 0.125 are raised to 0.125 (ffmpeg's xfade drops frames below that).",
        )
        MAX_RERENDER_SECONDS: int = Field(
            default=30, ge=4, le=30,
            description="Longest clip rerender_video will re-render in high fidelity (cost control; hifi with a video input costs ~2x per second).",
        )
        EMBED_SAVED_COPY: bool = Field(
            default=True,
            description="Play the copy saved in Open WebUI inside the chat embed via a signed link (needs the KeplerAI backend's kepler_files router). Falls back to the provider link if unavailable.",
        )
        SIGNED_LINK_TTL_DAYS: int = Field(default=30, ge=1, le=365, description="Lifetime of the signed embed link for saved clips.")
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
            "max_ref_videos": self.valves.MAX_REFERENCE_VIDEOS,
            "max_ref_audio": self.valves.MAX_REFERENCE_AUDIO,
            "poll_interval": self.valves.POLL_INTERVAL_SECONDS,
            "timeout": self.valves.GENERATION_TIMEOUT_SECONDS,
            "save_to_openwebui": self.valves.SAVE_TO_OPENWEBUI,
            "join_extensions": self.valves.JOIN_EXTENSIONS,
            "join_crossfade": float(self.valves.JOIN_CROSSFADE_SECONDS),
            "join_cut_search": float(self.valves.JOIN_CUT_SEARCH_SECONDS),
            "max_rerender": int(self.valves.MAX_RERENDER_SECONDS),
            "embed_saved_copy": self.valves.EMBED_SAVED_COPY,
            "signed_link_ttl": int(self.valves.SIGNED_LINK_TTL_DAYS) * 24 * 3600,
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
        """Turn a data URI, Open WebUI file reference, or public URL into a MediaRef (images/audio)."""
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

    @staticmethod
    def _seedance_meta(file: Any) -> Dict[str, Any]:
        meta = getattr(file, "meta", None) or {}
        sd = meta.get("seedance") or {}
        return sd if isinstance(sd, dict) else {}

    @staticmethod
    def _unexpired_provider_url(sd: Dict[str, Any]) -> Optional[str]:
        url = sd.get("provider_url")
        exp = sd.get("provider_url_expires_at") or 0
        if url and exp and time.time() < float(exp) - 120:
            return url
        return None

    async def _latest_generated_video(self, user_id: Optional[str], chat_id: Optional[str]) -> Optional[Any]:
        """Most recent clip this tool generated for the user, preferring the current chat."""
        if not user_id:
            return None
        try:
            from open_webui.models.files import Files
        except Exception:
            return None
        files = await Files.get_files_by_user_id(user_id)
        mine = [f for f in files if self._seedance_meta(f).get("provider_url") or (f.meta or {}).get("source") == "kepler_seedance_video"]
        if not mine:
            return None
        mine.sort(key=lambda f: f.created_at or 0, reverse=True)
        if chat_id:
            same_chat = [f for f in mine if self._seedance_meta(f).get("chat_id") == chat_id]
            if same_chat:
                return same_chat[0]
        return mine[0]

    async def _resolve_video_source(
        self, source: Optional[str], user_id: Optional[str], chat_id: Optional[str]
    ) -> Tuple[str, Dict[str, Any], Any]:
        """Return (public_url, seedance_meta, owui_file_or_None) for a source video. ModelArk only fetches
        public web URLs, so KeplerAI files are usable only via their stored provider URL (valid ~24h)."""
        s = (source or "").strip()
        file = None

        if not s:
            file = await self._latest_generated_video(user_id, chat_id)
            if not file:
                raise VideoGenError("No previous video found in this chat. Generate a clip first, or give a public https:// video URL.")
        else:
            m = OWUI_FILE_ID_RE.search(s)
            if m or UUID_RE.match(s):
                try:
                    from open_webui.models.files import Files
                except Exception as exc:
                    raise VideoGenError(f"Cannot access Open WebUI file storage: {exc}") from exc
                file = await Files.get_file_by_id(m.group(1) if m else s)
                if not file:
                    raise VideoGenError("That KeplerAI file was not found.")
            elif s.startswith("https://") or s.startswith("http://"):
                return s, {}, None
            else:
                raise VideoGenError("Unsupported video reference. Use the last generated clip, a KeplerAI file link, or a public https:// URL.")

        sd = self._seedance_meta(file)
        # Preferred: a signed KeplerAI link (works for stitched clips, old clips and user uploads, and it
        # is the whole saved file rather than just the newest segment). Verified 2026-09-07 that
        # ModelArk fetches these. Fallback: the provider's own link while it is still valid.
        signed = self._signed_public_url(file.id)
        if signed:
            return signed, sd, file
        url = self._unexpired_provider_url(sd)
        if not url:
            when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(file.created_at or 0))
            raise VideoGenError(
                f"The source clip (generated {when} UTC) can no longer be fetched by the video provider: "
                "ByteDance only accepts public web URLs and its own links expire after 24 hours. "
                "Regenerate the clip, or host the MP4 at a public https:// URL and pass that."
            )
        return url, sd, file

    @staticmethod
    def _signed_public_url(file_id: str, ttl_seconds: int = 6 * 3600) -> Optional[str]:
        """Absolute signed URL to a stored file for the video provider to fetch (needs WEBUI_URL)."""
        try:
            from open_webui.config import WEBUI_URL
            from open_webui.routers.kepler_files import sign_file_url
        except Exception:
            return None
        base = (WEBUI_URL or "").rstrip("/")
        if not base.startswith("https://"):
            return None
        return base + sign_file_url(file_id, ttl_seconds)

    # ------------------------------------------------------------------ ffmpeg join

    @staticmethod
    def _probe(ffmpeg: str, ffprobe: Optional[str], path: str) -> Dict[str, Any]:
        """Streams + duration. Uses ffprobe when present, else parses ffmpeg's banner."""
        info: Dict[str, Any] = {"has_audio": False, "width": 0, "height": 0, "duration": 0.0}
        if ffprobe:
            out = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json", path],
                capture_output=True, text=True, timeout=60,
            )
            try:
                data = json.loads(out.stdout or "{}")
                for st in data.get("streams", []):
                    if st.get("codec_type") == "video" and not info["width"]:
                        info["width"], info["height"] = int(st.get("width") or 0), int(st.get("height") or 0)
                    if st.get("codec_type") == "audio":
                        info["has_audio"] = True
                info["duration"] = float((data.get("format") or {}).get("duration") or 0)
                return info
            except (ValueError, TypeError):
                pass
        banner = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=60).stderr
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", banner)
        if m:
            info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
        m = re.search(r"Video: .*?(\d{2,5})x(\d{2,5})", banner)
        if m:
            info["width"], info["height"] = int(m[1]), int(m[2])
        info["has_audio"] = "Audio:" in banner
        return info

    @staticmethod
    def _gray_frames(ffmpeg: str, path: str, args: List[str], w: int = 160, h: int = 90, fps: int = 24) -> List[bytes]:
        """Decode frames as tiny 8-bit grayscale rasters for cheap pixel comparisons."""
        res = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", *args, "-i", path, "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, timeout=120,
        )
        n = w * h
        raw = res.stdout or b""
        return [raw[i:i + n] for i in range(0, len(raw) - n + 1, n)]

    @staticmethod
    def _mad(a: bytes, b: bytes) -> float:
        """Mean absolute difference of two equal-length gray rasters (0-255)."""
        if not a or not b or len(a) != len(b):
            return 255.0
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    @staticmethod
    def _vresample(frame: bytes, w: int, h: int, s: float) -> bytes:
        """Stretch a gray raster vertically by factor s about its centre (nearest neighbour), same size out."""
        fill = bytes([sum(frame) // len(frame)]) * w
        rows = []
        for y in range(h):
            sy = int(round((y - h / 2) / s + h / 2))
            rows.append(frame[sy * w:(sy + 1) * w] if 0 <= sy < h else fill)
        return b"".join(rows)

    def _best_vscale(self, ref: bytes, frame: bytes, w: int, h: int) -> Tuple[float, float, float]:
        """Vertical scale factor to apply to `frame` so it best matches `ref`.

        Seedance renders reference-based continuations at its own native aspect (~1.74:1) and stretches them
        to the requested 16:9 frame, so they arrive ~2% vertically compressed relative to the source clip.
        Returns (factor, mad_before, mad_after)."""
        before = self._mad(ref, frame)
        best_s, best_d = 1.0, before
        for i in range(-12, 17):  # 0.94 .. 1.08 in 0.005 steps
            s = 1.0 + i * 0.005
            d = self._mad(ref, self._vresample(frame, w, h, s))
            if d < best_d - 1e-6:
                best_s, best_d = s, d
        return round(best_s, 3), before, best_d

    def _best_cut_point(self, ffmpeg: str, a: str, b: str, window_s: float, fps: int = 24) -> Tuple[float, Dict[str, Any]]:
        """Where to start the continuation so it joins the source most seamlessly.

        Technique (as used by the community for Seedance extensions): compare every frame in the first
        `window_s` of the continuation with the source's final frame and pick the closest one, then skip
        the near-static frames the model tends to produce right after that point so motion resumes at once.
        Returns (start offset in seconds, debug info)."""
        last = self._gray_frames(ffmpeg, a, ["-sseof", "-0.15"])
        head = self._gray_frames(ffmpeg, b, ["-t", str(window_s)])
        if not last or len(head) < 3:
            return 0.0, {"reason": "no frames"}
        ref = last[-1]
        diffs = [self._mad(ref, f) for f in head]
        best = min(range(len(diffs)), key=lambda i: diffs[i])
        # Skip stalled frames after the match: advance while consecutive frames barely change.
        motion = [self._mad(head[i], head[i + 1]) for i in range(len(head) - 1)]
        typical = sorted(motion)[len(motion) // 2] if motion else 0.0
        stall_eps = max(0.35, typical * 0.35)
        j = best
        while j < len(motion) and motion[j] < stall_eps and j - best < fps:  # never skip more than 1s
            j += 1
        return round(j / fps, 3), {"best_match_frame": best, "match_diff": round(diffs[best], 2), "start_frame": j, "typical_motion": round(typical, 2)}

    def _concat_videos(self, first: bytes, second: bytes, crossfade: float = 0.0, cut_search: float = 0.0) -> Tuple[bytes, float]:
        """Stitch two MP4s back to back (re-encoded so mismatched params never break the join).
        cut_search > 0: trim the continuation to its best-matching, motion-resumed frame first.
        crossfade > 0: short dissolve at the seam. Returns (mp4 bytes, total duration seconds)."""
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg:
            raise VideoGenError("ffmpeg is not available on the server")
        with tempfile.TemporaryDirectory(prefix="kepler_seedance_") as tmp:
            a, b, out = os.path.join(tmp, "a.mp4"), os.path.join(tmp, "b.mp4"), os.path.join(tmp, "joined.mp4")
            with open(a, "wb") as fh:
                fh.write(first)
            with open(b, "wb") as fh:
                fh.write(second)
            pa, pb = self._probe(ffmpeg, ffprobe, a), self._probe(ffmpeg, ffprobe, b)
            w, h = (pa["width"] or pb["width"] or 1280), (pa["height"] or pb["height"] or 720)
            with_audio = pa["has_audio"] and pb["has_audio"]
            da, db = float(pa["duration"] or 0), float(pb["duration"] or 0)

            cut = 0.0
            if cut_search and cut_search > 0.1 and db > cut_search + 1.0:
                try:
                    cut, info = self._best_cut_point(ffmpeg, a, b, cut_search)
                    log.info("kepler_seedance_video: stitch cut point %.3fs into continuation (%s)", cut, info)
                except Exception as exc:
                    log.warning("kepler_seedance_video: cut-point search failed (%s); joining at frame 0", exc)
                    cut = 0.0
            db_eff = max(db - cut, 0.5)
            self._last_cut = cut

            # Geometry match: the continuation usually arrives slightly vertically compressed (see
            # _best_vscale). Measure it against the source's last frame and stretch it back, cropping the
            # sliver that overflows, so the seam has no size jump and the rest of the clip matches too.
            geom = ""
            try:
                gw, gh = 320, 180
                ref = self._gray_frames(ffmpeg, a, ["-sseof", "-0.15"], gw, gh)[-1]
                probe_frames = self._gray_frames(ffmpeg, b, ["-ss", str(cut), "-t", "0.2"], gw, gh)
                if probe_frames:
                    vs, d0, d1 = self._best_vscale(ref, probe_frames[0], gw, gh)
                    log.info("kepler_seedance_video: stitch geometry vscale=%.3f (frame diff %.2f -> %.2f)", vs, d0, d1)
                    if abs(vs - 1.0) >= 0.0075:
                        if vs > 1:
                            geom = f"scale={w}:{int(round(h * vs))}:flags=lanczos,crop={w}:{h},"
                        else:
                            geom = f"scale={int(round(w / vs))}:{h}:flags=lanczos,crop={w}:{h},"
            except Exception as exc:
                log.warning("kepler_seedance_video: geometry match skipped (%s)", exc)

            trim_v = f"trim=start={cut},setpts=PTS-STARTPTS," if cut > 0 else ""
            trim_a = f"atrim=start={cut},asetpts=PTS-STARTPTS," if cut > 0 else ""
            scale = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24"
            # The continuation is stretched to the source frame exactly as it was measured, then geometry-fixed.
            scale_b = f"scale={w}:{h}:flags=lanczos,{geom}setsar=1,fps=24"
            xf = min(float(crossfade or 0), 2.0)
            use_xfade = xf > 0.03 and da > xf + 0.5 and db_eff > xf + 0.5
            if use_xfade and xf < 0.125:
                xf = 0.125  # measured: xfade shorter than 3 frames at 24fps halves the output frame rate
            if use_xfade:
                off = round(da - xf, 3)
                total = da + db_eff - xf
                if with_audio:
                    fc = (f"[0:v]{scale}[v0];[1:v]{trim_v}{scale_b}[v1];[v0][v1]xfade=transition=fade:duration={xf}:offset={off}[v];"
                          f"[0:a]aresample=48000[a0];[1:a]{trim_a}aresample=48000[a1];[a0][a1]acrossfade=d={xf}[a]")
                    maps = ["-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
                else:
                    fc = f"[0:v]{scale}[v0];[1:v]{trim_v}{scale_b}[v1];[v0][v1]xfade=transition=fade:duration={xf}:offset={off}[v]"
                    maps = ["-map", "[v]", "-an"]
            else:
                total = da + db_eff
                if with_audio:
                    fc = (f"[0:v]{scale}[v0];[1:v]{trim_v}{scale_b}[v1];[0:a]aresample=48000[a0];[1:a]{trim_a}aresample=48000[a1];"
                          f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
                    maps = ["-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
                else:
                    fc = f"[0:v]{scale}[v0];[1:v]{trim_v}{scale_b}[v1];[v0][v1]concat=n=2:v=1:a=0[v]"
                    maps = ["-map", "[v]", "-an"]
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", a, "-i", b, "-filter_complex", fc, *maps,
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode != 0 or not os.path.exists(out):
                raise VideoGenError(f"ffmpeg concat failed: {(res.stderr or '').strip()[-300:]}")
            with open(out, "rb") as fh:
                return fh.read(), total

    @staticmethod
    def _last_frame_jpeg(video: bytes) -> bytes:
        """Grab the final frame of an MP4 as JPEG bytes (used to anchor continuations)."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise VideoGenError("ffmpeg is not available on the server")
        with tempfile.TemporaryDirectory(prefix="kepler_seedance_") as tmp:
            src, out = os.path.join(tmp, "src.mp4"), os.path.join(tmp, "last.jpg")
            with open(src, "wb") as fh:
                fh.write(video)
            res = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-sseof", "-0.1", "-i", src, "-frames:v", "1", "-q:v", "2", out],
                capture_output=True, text=True, timeout=120,
            )
            if res.returncode != 0 or not os.path.exists(out):
                raise VideoGenError(f"ffmpeg last-frame extraction failed: {(res.stderr or '').strip()[-200:]}")
            with open(out, "rb") as fh:
                return fh.read()

    async def _read_file_bytes(self, file: Any, fallback_url: Optional[str]) -> bytes:
        if file is not None and getattr(file, "path", None):
            try:
                from open_webui.storage.provider import Storage
                local_path = await asyncio.to_thread(Storage.get_file, file.path)
                with open(local_path, "rb") as fh:
                    return fh.read()
            except Exception as exc:
                log.warning("kepler_seedance_video: reading stored source failed (%s); downloading instead", exc)
        if not fallback_url:
            raise VideoGenError("Source video bytes unavailable")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            async with session.get(fallback_url) as resp:
                if resp.status != 200:
                    raise VideoGenError(f"Could not download source video ({resp.status})")
                return await resp.read()

    def _collect_media(
        self,
        messages: Optional[List[Dict[str, Any]]],
        files: Optional[List[Dict[str, Any]]],
        media_type: str,
    ) -> List[str]:
        """Collect image/audio/video references from __files__ and the latest user message that has any."""
        found: List[str] = []

        def match(f: Dict[str, Any]) -> Optional[str]:
            ctype = ((f.get("file") or {}).get("meta") or {}).get("content_type") or ""
            ftype = f.get("type") or ""
            ok = (
                (media_type == "image" and (ftype == "image" or ctype.startswith("image/")))
                or (media_type == "audio" and ctype.startswith("audio/"))
                or (media_type == "video" and (ftype == "video" or ctype.startswith("video/")))
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
        ref_videos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Submit a ModelArk video generation task; returns {url, tokens, task_id, seed}."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if first_frame:
            content.append({"type": "image_url", "image_url": {"url": first_frame.as_data_uri()}, "role": "first_frame"})
        for ref in ref_images or []:
            content.append({"type": "image_url", "image_url": {"url": ref.as_data_uri()}, "role": "reference_image"})
        for url in ref_videos or []:
            content.append({"type": "video_url", "video_url": {"url": url}, "role": "reference_video"})
        for ref in ref_audios or []:
            content.append({"type": "audio_url", "audio_url": {"url": ref.as_data_uri()}, "role": "reference_audio"})

        payload: Dict[str, Any] = {"model": model, "content": content, "watermark": False, **params}
        # ModelArk only understands 'adaptive' when there is visual input to adapt to.
        if payload.get("ratio") == "adaptive" and not (first_frame or ref_images or ref_videos):
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
        ref_videos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {config['api_key']}"}
        base = config["atlas_base_url"]
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config["timeout"] + 60), headers=headers) as session:
            payload: Dict[str, Any] = {"model": model, "prompt": prompt, **params}
            if first_frame:
                payload["image_url"] = await self._atlas_upload(session, base, first_frame)
            if ref_images:
                payload["reference_images"] = [await self._atlas_upload(session, base, r) for r in ref_images]
            if ref_videos:
                payload["reference_videos"] = list(ref_videos)
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

    async def _download(self, url: str) -> Tuple[bytes, str]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise VideoGenError(f"Could not download finished video ({resp.status}).")
                data = await resp.read()
                return data, (resp.headers.get("Content-Type", "video/mp4").split(";")[0] or "video/mp4")

    @staticmethod
    def _signed_embed_url(file_id: str, ttl_seconds: int) -> Optional[str]:
        """Signed, session-free link served by the KeplerAI backend (routers/kepler_files.py)."""
        try:
            from open_webui.routers.kepler_files import sign_file_url
        except Exception:
            return None
        try:
            return sign_file_url(file_id, ttl_seconds)
        except Exception:
            log.exception("kepler_seedance_video: signing embed link failed")
            return None

    async def _save_bytes_to_openwebui(
        self, data: bytes, content_type: str, user_id: Optional[str], filename: str, extra_meta: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, str]]:
        """Register bytes as an Open WebUI file. Returns (permanent relative URL, file id)."""
        if not user_id:
            return None
        try:
            from open_webui.models.files import FileForm, Files
            from open_webui.storage.provider import Storage
        except Exception:
            return None

        file_id = str(uuid.uuid4())
        stored_name = f"{file_id}_{filename}"
        _, path = await asyncio.to_thread(
            Storage.upload_file, io.BytesIO(data), stored_name, {"OpenWebUI-User-Id": user_id, "OpenWebUI-File-Id": file_id}
        )
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
        return f"/api/v1/files/{file_id}/content", file_id

    def _render(
        self,
        provider_url: str,
        saved_url: Optional[str],
        config: Dict[str, Any],
        summary: str,
        save_error: Optional[str] = None,
        usage_note: str = "",
        ratio: str = "16:9",
        embed_url: Optional[str] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        embed_url = embed_url or provider_url
        if saved_url:
            keep_note = (
                f"A permanent copy is saved in KeplerAI at {saved_url}. It can be extended, edited or re-rendered in "
                "high fidelity later by just asking (e.g. 'extend it', 'make the boat red', 'make a hifi version')."
            )
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
            # The chat measures this document's height once on load, before the <video> knows its
            # dimensions, which yields a sliver. Reserve the box with aspect-ratio up front and also
            # report our height to the parent (FullHeightIframe listens for 'iframe:height').
            w, h = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9)}.get(
                ratio, (16, 9)
            )
            max_h = 540 if h <= w else 640
            html = (
                '<style>html,body{margin:0;padding:0;background:transparent}'
                f'.wrap{{width:100%;max-width:{int(max_h * w / h)}px;aspect-ratio:{w}/{h};max-height:{max_h}px;margin:0 auto}}'
                '.wrap video{width:100%;height:100%;display:block;object-fit:contain;background:#000;border-radius:10px}'
                '.meta{font-family:system-ui,sans-serif;font-size:12px;margin:8px 4px 0;line-height:1.5}'
                '.meta a{color:#3b82f6;text-decoration:none}.meta a:hover{text-decoration:underline}</style>'
                f'<div id="root"><div class="wrap"><video id="v" controls autoplay muted playsinline preload="metadata" src="{embed_url}"></video></div>'
                f'<p class="meta">{links}{usage_html}</p></div>'
                # Measure the content wrapper, never the document: documentElement.scrollHeight is
                # floored at the viewport height, so posting it back after each resize inflates the
                # frame forever (that was the blank-space bug). Only post when the value changes.
                "<script>(function(){var last=0;function r(){try{var h=Math.ceil(document.getElementById('root').getBoundingClientRect().height)+4;"
                "if(Math.abs(h-last)>2){last=h;parent.postMessage({type:'iframe:height',height:h},'*')}}catch(e){}}"
                "var v=document.getElementById('v');v.addEventListener('loadedmetadata',r);"
                "window.addEventListener('load',r);window.addEventListener('resize',r);setTimeout(r,300);setTimeout(r,1500)})();</script>"
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
        ref_videos: Optional[List[str]] = None,
        chat_id: Optional[str] = None,
        source_url: Optional[str] = None,
        join_source: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        if not config["api_key"]:
            which = "BYTEPLUS_API_KEY" if config["provider"] == "byteplus" else "ATLASCLOUD_API_KEY"
            return f"Video generation is not configured yet: an admin must set {which} in the tool's Valves."

        # Quality tier -> model. Multi-reference mode needs Seedance 2.5, so it always runs on 'hifi'.
        tier = TIER_ALIASES.get(str(quality or "").lower().strip(), config["default_quality"])
        if mode in ("reference", "rerender"):
            tier = "hifi"
        if config["provider"] == "byteplus":
            # ModelArk uses one model id for text / first-frame / reference / edit / extend inputs.
            model = config["byteplus_models"][tier]
            runner = self._byteplus_generate
        else:
            model = {
                "text": config["atlas_best_model"] if tier == "hifi" else config["atlas_fast_model"],
                "image": config["atlas_image_to_video_model"],
                "reference": config["atlas_reference_model"],
                "extend": config["atlas_reference_model"],
                "edit": config["atlas_reference_model"],
                "rerender": config["atlas_reference_model"],
            }[mode]
            runner = self._atlas_generate

        # API limits per model family: Seedance 2.0 (draft/mini) tops out at 15s, 2.5 at 30s.
        model_max = 30 if tier == "hifi" else 15
        if params["duration"] > model_max:
            await self._emit_status(emitter, f"{model} supports at most {model_max}s per clip; using {model_max}s", done=False)
            params["duration"] = model_max

        # Wall-time estimate for the progress bar (the API gives no percentage).
        rate = EST_SECONDS_PER_VIDEO_SECOND.get(params["resolution"], 25.0)
        audio_factor = 1.15 if params.get("generate_audio") else 1.0
        video_factor = EST_VIDEO_INPUT_FACTOR if ref_videos else 1.0
        config["est_total"] = EST_OVERHEAD_SECONDS + params["duration"] * rate * EST_MODEL_FACTOR[tier] * audio_factor * video_factor

        await self._emit_status(
            emitter, f"Submitting to {model} ({config['provider']}, ~{int(config['est_total'])}s estimated)", done=False
        )
        try:
            result = await runner(config, emitter, label, model, prompt, params, first_frame, ref_images, ref_audios, ref_videos)
        except (VideoGenError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("kepler_seedance_video: generation failed: %s", exc)
            await self._emit_status(emitter, f"Video generation error: {exc}", done=True)
            return f"Video generation failed: {exc}"

        provider_url = result["url"]
        tokens = int(result.get("tokens") or 0)
        usage_note = ""
        if tokens:
            price = float(config["prices"].get(tier) or 0)
            if price > 0:
                # Cost first; the raw token count stays available in the file metadata for reporting.
                usage_note = f"≈ ${tokens / 1_000_000 * price:.2f} this clip ({params['duration']}s {params['resolution']}, {tier})"
            else:
                usage_note = f"{tokens:,} video tokens ({params['duration']}s {params['resolution']}, {tier})"
        log.info(
            "kepler_seedance_video: %s mode=%s tier=%s model=%s duration=%ss res=%s tokens=%s task=%s user=%s chat=%s",
            config["provider"], mode, tier, model, params["duration"], params["resolution"], tokens,
            result.get("task_id"), (user or {}).get("id"), chat_id,
        )

        saved_url, save_error, embed_url = None, None, None
        joined, total_duration = False, params["duration"]
        if config["save_to_openwebui"]:
            await self._emit_status(emitter, "Saving video to KeplerAI storage", done=False)
            try:
                data, content_type = await self._download(provider_url)
                if join_source and config["join_extensions"]:
                    await self._emit_status(emitter, "Stitching source and continuation into one clip", done=False)
                    try:
                        src_bytes = join_source.get("bytes") or await self._read_file_bytes(join_source.get("file"), join_source.get("url"))
                        data, joined_secs = await asyncio.to_thread(
                            self._concat_videos, src_bytes, data, config["join_crossfade"], config["join_cut_search"]
                        )
                        content_type = "video/mp4"
                        joined = True
                        total_duration = int(round(joined_secs)) if joined_secs else int(join_source.get("duration") or 0) + params["duration"]
                    except Exception as exc:
                        log.exception("kepler_seedance_video: stitching extension failed")
                        await self._emit_status(emitter, f"Could not stitch clips ({exc}); saving the continuation only", done=False)
                src_file = join_source.get("file") if join_source else None
                saved = await self._save_bytes_to_openwebui(
                    data,
                    content_type,
                    (user or {}).get("id"),
                    f"seedance-{time.strftime('%Y%m%d-%H%M%S')}.mp4",
                    {
                        "seedance": {
                            "provider": config["provider"],
                            "model": model,
                            "tier": tier,
                            "mode": mode,
                            "task_id": result.get("task_id"),
                            "seed": result.get("seed"),
                            "tokens": tokens,
                            "prompt": prompt,
                            "duration": total_duration,
                            "segment_duration": params["duration"],
                            "joined": joined,
                            "joined_from": getattr(src_file, "id", None) if joined else None,
                            "resolution": params["resolution"],
                            "ratio": params["ratio"],
                            "chat_id": chat_id,
                            "source_video": source_url,
                            # Always the provider link of the NEWEST segment: that is what a further
                            # extension must continue from, and the only URL ByteDance can fetch.
                            "provider_url": provider_url,
                            "provider_url_expires_at": int(time.time()) + PROVIDER_URL_TTL_SECONDS,
                        }
                    },
                )
                saved_url, saved_id = saved if saved else (None, None)
                if not saved_url:
                    save_error = "storage layer unavailable or no user id"
                elif config["embed_saved_copy"]:
                    # Session-free signed link so the sandboxed chat embed can play the saved (possibly
                    # stitched) copy; None when the backend lacks the kepler_files router.
                    embed_url = self._signed_embed_url(saved_id, config["signed_link_ttl"])
            except Exception as exc:  # saving is best-effort; the provider link still works
                save_error = str(exc)
                log.exception("kepler_seedance_video: saving video into Open WebUI storage failed")

        await self._emit_status(emitter, f"{label}: done", done=True)
        if mode == "extend" and joined:
            player_note = (
                "The player and the saved copy show the full joined video."
                if embed_url
                else "The saved copy in KeplerAI is the full joined video; the player shows only the new segment."
            )
            summary = (
                f"Extended the clip to {total_duration}s total (original + {params['duration']}s continuation). {player_note} "
                f"{params['resolution']}, {model} ({tier} tier) via {config['provider']}. "
                "Note for the user: a stitched extension always has a soft seam because the provider cannot continue "
                "motion frame-exactly; for a seamless final, they can ask for a hifi version, which re-renders the whole "
                "clip as one continuous take."
            )
        else:
            verb = {
                "text": "Generated",
                "image": "Animated the attached image into",
                "reference": "Generated from the references",
                "extend": "Extended the source clip into (continuation segment only)",
                "edit": "Edited the source clip into",
                "rerender": "Re-rendered the source clip in high fidelity as",
            }.get(mode, "Generated")
            summary = f"{verb} a {params['duration']}s {params['resolution']} clip with {model} ({tier} tier) via {config['provider']}."
        return self._render(provider_url, saved_url, config, summary, save_error, usage_note, params.get("ratio") or "16:9", embed_url)

    async def _prepare_refs(self, emitter: EventEmitter, refs: List[str], kind: str) -> List[MediaRef]:
        if refs:
            await self._emit_status(emitter, f"Preparing {len(refs)} {kind} reference(s)", done=False)
        return [await self._resolve_media(r) for r in refs]

    async def _prepare_video_refs(
        self, emitter: EventEmitter, refs: List[str], user_id: Optional[str], chat_id: Optional[str]
    ) -> List[str]:
        out: List[str] = []
        if refs:
            await self._emit_status(emitter, f"Resolving {len(refs)} video reference(s)", done=False)
        for r in refs:
            url, _, _ = await self._resolve_video_source(r, user_id, chat_id)
            out.append(url)
        return out

    @staticmethod
    def _chat_id(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        return (metadata or {}).get("chat_id") if isinstance(metadata, dict) else None

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
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate a NEW short video clip from a text prompt using ByteDance Seedance. Use this when the user describes
        a scene or asks for a video, clip, animation or motion concept and has NOT attached an image, and is NOT
        asking to change or continue an existing clip (use edit_video / extend_video for that).

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
        return await self._generate(
            config, __event_emitter__, __user__, "Rendering video", quality, "text", prompt, params,
            chat_id=self._chat_id(__metadata__),
        )

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
        __metadata__: Optional[Dict[str, Any]] = None,
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
            config, __event_emitter__, __user__, "Animating image", quality, "image", prompt, params,
            first_frame=first_frame, chat_id=self._chat_id(__metadata__),
        )

    async def generate_video_from_references(
        self,
        prompt: str,
        image_urls: Optional[List[str]] = None,
        video_urls: Optional[List[str]] = None,
        audio_urls: Optional[List[str]] = None,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "16:9",
        generate_audio: bool = True,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate a NEW video guided by SEVERAL reference assets - characters, props, environments, style frames,
        audio, or a video whose camera/motion/style should be matched. Use this when the user attached two or more
        images, an audio clip, or wants a previous clip used as a style/motion reference for a different scene.
        References are picked up from the chat automatically. In the prompt, cite them in order as
        "Image 1", "Image 2", "Video 1", "Audio 1", e.g. "Image 1 walks through the market from Image 2, matching
        the camera movement of Video 1". Always uses the high-fidelity model.

        :param prompt: Shot description citing the references as Image 1 / Video 1 / Audio 1.
        :param image_urls: Optional explicit public https:// image URLs. Leave empty to use attachments.
        :param video_urls: Optional reference videos: public https:// URLs, KeplerAI file links, or "last" for the most recent clip generated in this chat. Leave empty to use attachments.
        :param audio_urls: Optional explicit public https:// audio URLs. Leave empty to use attachments.
        :param duration: Clip length in seconds (4-10). Default 5.
        :param resolution: 480p, 720p or 1080p. Default 720p.
        :param ratio: Aspect ratio: 16:9 (default), 9:16, 1:1, 4:3, 3:4, 21:9 or adaptive.
        :param generate_audio: Whether to generate synchronised sound. Default true.
        """
        config = self._resolve_config(__user__)
        user_id, chat_id = (__user__ or {}).get("id"), self._chat_id(__metadata__)
        images = (list(image_urls or []) or self._collect_media(__messages__, __files__, "image"))[: config["max_ref_images"]]
        videos = (list(video_urls or []) or self._collect_media(__messages__, __files__, "video"))[: config["max_ref_videos"]]
        audios = (list(audio_urls or []) or self._collect_media(__messages__, __files__, "audio"))[: config["max_ref_audio"]]
        videos = ["" if str(v).strip().lower() in ("last", "latest", "previous") else v for v in videos]
        if not images and not audios and not videos:
            return "No reference images, videos or audio found. Ask the user to attach them, then call this tool again."
        try:
            ref_images = await self._prepare_refs(__event_emitter__, images, "image")
            ref_audios = await self._prepare_refs(__event_emitter__, audios, "audio")
            ref_videos = await self._prepare_video_refs(__event_emitter__, videos, user_id, chat_id)
        except (VideoGenError, OSError) as exc:
            return f"Could not prepare the reference assets: {exc}"
        params = {
            "duration": self._clamp_duration(duration, config),
            "resolution": self._clamp_resolution(resolution, config),
            "ratio": self._clamp_ratio(ratio),
            "generate_audio": bool(generate_audio),
        }
        return await self._generate(
            config, __event_emitter__, __user__, "Rendering from references", "hifi", "reference", prompt, params,
            ref_images=ref_images, ref_audios=ref_audios, ref_videos=ref_videos, chat_id=chat_id,
            source_url=ref_videos[0] if ref_videos else None,
        )

    async def extend_video(
        self,
        prompt: str,
        source_video: str = "",
        duration: int = 5,
        resolution: str = "",
        generate_audio: bool = True,
        quality: str = "",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        CONTINUE an existing clip: generates the next few seconds after the source video ends, keeping the same
        subject, setting, camera and style. Use when the user says things like "extend it", "continue the video",
        "what happens next", "make it longer". By default the source is the most recent clip generated in this chat,
        so you usually do not need to pass source_video. The saved copy in KeplerAI is the full joined video
        (original followed by the continuation); the inline player shows only the new segment.

        :param prompt: What should happen next (e.g. "the boat reaches the edge of the puddle and stops"). Write it as a continuation; the tool adds the technical framing.
        :param source_video: Leave empty for the last clip generated in this chat. Otherwise a KeplerAI file link or a public https:// video URL. Clips older than 24 hours cannot be used.
        :param duration: Length of the continuation in seconds (4-10). Default 5.
        :param resolution: Leave empty to match the source clip; or 480p, 720p, 1080p.
        :param generate_audio: Whether to generate synchronised sound. Default true.
        :param quality: Leave empty for the configured default; "draft" or "hifi".
        """
        config = self._resolve_config(__user__)
        user_id, chat_id = (__user__ or {}).get("id"), self._chat_id(__metadata__)
        src = source_video or next(iter(self._collect_media(__messages__, __files__, "video")), "")
        try:
            src_url, src_meta, src_file = await self._resolve_video_source(src, user_id, chat_id)
        except VideoGenError as exc:
            return f"Cannot extend: {exc}"

        # Anchor the continuation on the source's final frame. Tested 2026-09-07: a reference video
        # alone gives a thematic continuation with a visible jump at the cut; adding the last frame as
        # a reference image makes the first frames line up almost exactly (ModelArk refuses a
        # first_frame role alongside reference media, so reference_image is the only way).
        src_bytes: Optional[bytes] = None
        ref_images: List[MediaRef] = []
        try:
            await self._emit_status(__event_emitter__, "Reading the source clip's last frame", done=False)
            src_bytes = await self._read_file_bytes(src_file, src_url)
            jpg = await asyncio.to_thread(self._last_frame_jpeg, src_bytes)
            ref_images = [MediaRef(data=jpg, mime="image/jpeg", name="last_frame.jpg")]
        except Exception as exc:
            log.warning("kepler_seedance_video: could not extract last frame (%s); extending from the video alone", exc)

        text = prompt.strip().rstrip(".")
        if not VIDEO1_RE.search(text):
            if ref_images:
                text = (
                    f"Image 1 is the exact last frame of Video 1. Continue seamlessly from Image 1: {text}. "
                    "Keep the same subject, setting, camera angle, lighting and visual style as Video 1."
                )
            else:
                text = (
                    f"Extend Video 1 forward: {text}. Keep the same subject, setting, camera angle, "
                    "lighting and visual style as Video 1 so the continuation cuts together seamlessly."
                )
        params = {
            "duration": self._clamp_duration(duration, config),
            "resolution": self._clamp_resolution(resolution or src_meta.get("resolution"), config),
            "ratio": self._clamp_ratio(src_meta.get("ratio") or "adaptive"),
            "generate_audio": bool(generate_audio),
        }
        return await self._generate(
            config, __event_emitter__, __user__, "Extending video", quality or src_meta.get("tier") or "", "extend", text, params,
            ref_images=ref_images, ref_videos=[src_url], chat_id=chat_id, source_url=src_url,
            join_source={"url": src_url, "file": src_file, "bytes": src_bytes, "duration": src_meta.get("duration")},
        )

    def _video_duration(self, data: bytes) -> float:
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg:
            return 0.0
        with tempfile.TemporaryDirectory(prefix="kepler_seedance_") as tmp:
            p = os.path.join(tmp, "src.mp4")
            with open(p, "wb") as fh:
                fh.write(data)
            return float(self._probe(ffmpeg, ffprobe, p).get("duration") or 0.0)

    async def rerender_video(
        self,
        prompt: str = "",
        source_video: str = "",
        resolution: str = "1080p",
        generate_audio: bool = True,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        RE-RENDER an existing clip in HIGH FIDELITY (Seedance 2.5) keeping the same shot: same subject, composition,
        camera, motion and timing, just rendered with more detail and at higher resolution. Use when the user is happy
        with a draft and asks for a "hifi", "high quality", "final", "polished", "upscaled" or "production" version of
        it. By default the source is the most recent clip in this chat (including stitched extensions). This always
        uses the high-fidelity model and costs several times a draft clip.

        :param prompt: Optional small adjustments to apply while re-rendering (e.g. "slightly warmer light"). Leave empty to reproduce the draft faithfully.
        :param source_video: Leave empty for the last clip in this chat. Otherwise a KeplerAI file link or a public https:// video URL.
        :param resolution: Output resolution, default 1080p. 720p is cheaper.
        :param generate_audio: Whether to generate synchronised sound. Default true.
        """
        config = self._resolve_config(__user__)
        user_id, chat_id = (__user__ or {}).get("id"), self._chat_id(__metadata__)
        src = source_video or next(iter(self._collect_media(__messages__, __files__, "video")), "")
        try:
            src_url, src_meta, src_file = await self._resolve_video_source(src, user_id, chat_id)
        except VideoGenError as exc:
            return f"Cannot re-render: {exc}"

        src_duration = float(src_meta.get("duration") or 0)
        if not src_duration:
            try:
                src_duration = await asyncio.to_thread(self._video_duration, await self._read_file_bytes(src_file, src_url))
            except Exception as exc:
                log.warning("kepler_seedance_video: could not probe source duration (%s)", exc)
        duration = int(round(src_duration)) if src_duration else config["default_duration"]
        duration = min(max(duration, 4), config["max_rerender"])
        if src_duration and duration < int(src_duration) - 1:
            await self._emit_status(
                __event_emitter__, f"Source is {int(src_duration)}s; re-rendering the first {duration}s (MAX_RERENDER_SECONDS cap)", done=False
            )

        adjust = prompt.strip().rstrip(".")
        text = (
            "Recreate Video 1 exactly: the same subject, scene, composition, camera angle, framing, motion, timing and "
            "lighting. Do not change the content. Render it at higher visual fidelity with finer detail, cleaner "
            "textures and more realistic materials, reflections and lighting."
        )
        if adjust:
            text += f" Adjustments: {adjust}."
        params = {
            "duration": duration,
            "resolution": self._clamp_resolution(resolution or "1080p", config),
            "ratio": self._clamp_ratio(src_meta.get("ratio") or "adaptive"),
            "generate_audio": bool(generate_audio),
        }
        return await self._generate(
            config, __event_emitter__, __user__, "Re-rendering in high fidelity", "hifi", "rerender", text, params,
            ref_videos=[src_url], chat_id=chat_id, source_url=src_url,
        )

    async def edit_video(
        self,
        prompt: str,
        source_video: str = "",
        resolution: str = "",
        generate_audio: bool = True,
        quality: str = "",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        CHANGE something in an existing clip while keeping everything else: swap or recolour an object, change the
        weather, time of day, outfit, background, etc. Use when the user asks to modify, tweak, replace or change
        part of a video they already generated ("make the boat red", "make it night time", "remove the leaf").
        By default the source is the most recent clip generated in this chat. The output has the same length,
        framing and motion as the source.

        :param prompt: The change to make, e.g. "replace the white paper boat with a red one". The tool adds the keep-everything-else framing.
        :param source_video: Leave empty for the last clip generated in this chat. Otherwise a KeplerAI file link or a public https:// video URL. Clips older than 24 hours cannot be used.
        :param resolution: Leave empty to match the source clip; or 480p, 720p, 1080p.
        :param generate_audio: Whether to generate synchronised sound. Default true.
        :param quality: Leave empty for the configured default; "draft" or "hifi".
        """
        config = self._resolve_config(__user__)
        user_id, chat_id = (__user__ or {}).get("id"), self._chat_id(__metadata__)
        src = source_video or next(iter(self._collect_media(__messages__, __files__, "video")), "")
        try:
            src_url, src_meta, _ = await self._resolve_video_source(src, user_id, chat_id)
        except VideoGenError as exc:
            return f"Cannot edit: {exc}"
        text = prompt.strip()
        if not VIDEO1_RE.search(text):
            text = (
                f"In Video 1, {text[0].lower() + text[1:] if text else text}".rstrip(".")
                + ". Keep the camera, framing, motion, timing, lighting and everything else exactly the same as Video 1."
            )
        params = {
            "duration": self._clamp_duration(src_meta.get("duration") or config["default_duration"], config),
            "resolution": self._clamp_resolution(resolution or src_meta.get("resolution"), config),
            "ratio": self._clamp_ratio(src_meta.get("ratio") or "adaptive"),
            "generate_audio": bool(generate_audio),
        }
        return await self._generate(
            config, __event_emitter__, __user__, "Editing video", quality or src_meta.get("tier") or "", "edit", text, params,
            ref_videos=[src_url], chat_id=chat_id, source_url=src_url,
        )
