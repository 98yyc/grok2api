"""
Grok image edit service.
"""

import asyncio
import random
import re
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterable, List, Union, Any, Callable

import orjson
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError

from app.core.config import get_config
from app.core.exceptions import (
    AppException,
    ErrorType,
    UpstreamException,
    StreamIdleTimeoutError,
)
from app.core.logger import logger
from app.services.grok.utils.process import (
    BaseProcessor,
    _with_idle_timeout,
    _normalize_line,
    _collect_images,
    _is_http2_error,
)
from app.services.grok.utils.upload import UploadService
from app.services.grok.utils.retry import pick_token, rate_limited
from app.services.grok.services.chat import GrokChatService
from app.services.grok.services.video import VideoService
from app.services.grok.utils.stream import wrap_stream_with_usage
from app.services.reverse.media_post import MediaPostReverse
from app.services.token import EffortType


@dataclass
class ImageEditResult:
    stream: bool
    data: Union[AsyncGenerator[str, None], List[str]]
    token_used: str = ""


def _extract_image_post_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for pattern in (
        r"/generated/([0-9a-fA-F-]{32,36})(?:/|$)",
        r"/users/[^/]+/([0-9a-fA-F-]{32,36})(?:/content|/|$)",
        r"/imagine-public/(?:share-images|images)/([0-9a-fA-F-]{32,36})(?:\.[a-z]+|/|$)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    matches = re.findall(r"([0-9a-fA-F-]{32,36})", text)
    return matches[-1] if matches else ""


async def _try_log_image_share_link(
    token: str,
    post_id: str,
    *,
    local_url: str = "",
) -> None:
    token_text = str(token or "").strip()
    post_text = str(post_id or "").strip()
    if not token_text or not post_text:
        return
    try:
        logger.info(f"Image create-link attempt: post_id={post_text}")
        async with AsyncSession() as session:
            metadata = await MediaPostReverse.capture_metadata(
                session,
                token_text,
                post_text,
                media_type="image",
                local_url=local_url,
            )
        share_link = str(metadata.get("share_link") or "").strip()
        metadata_path = str(metadata.get("metadata_path") or "").strip()
        if share_link:
            logger.info(
                "Image create-link success: "
                f"post_id={post_text}, share_link={share_link}, metadata_path={metadata_path or '-'}"
            )
        else:
            logger.info(
                "Image create-link completed without shareLink: "
                f"post_id={post_text}, metadata_path={metadata_path or '-'}"
            )
    except Exception as e:
        details = getattr(e, "details", None)
        logger.warning(
            "Image create-link failed: "
            f"post_id={post_text}, error={e}, details={details}"
        )


def _is_upload_rejected_error(exc: Exception) -> bool:
    """判断是否为上游审核导致的上传拒绝。"""
    msg = str(exc or "").lower()
    if "content moderated" in msg or "content-moderated" in msg:
        return True
    if '"code":3' in msg or "'code': 3" in msg:
        return True

    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        status = details.get("status")
        body = str(details.get("body") or "").lower()
        err = str(details.get("error") or "").lower()
        if "content moderated" in body or "content-moderated" in body:
            return True
        if '"code":3' in body or "'code': 3" in body:
            return True
        # 某些链路只返回 400 + '"code"' 关键词，按拒绝处理。
        if status == 400 and ('"code"' in err or "moderated" in err):
            return True

    return False


def _is_upload_network_error(exc: Exception) -> bool:
    """判断是否为网络连通/网关挑战类上传失败。"""
    msg = str(exc or "").lower()
    if (
        "tls connect error" in msg
        or "timed out" in msg
        or "timeout" in msg
        or "connection" in msg
        or "proxy" in msg
        or "curl: (35)" in msg
    ):
        return True

    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        status = details.get("status")
        body = str(details.get("body") or "").lower()
        if status == 403 and ("just a moment" in body or "cloudflare" in body):
            return True
        if "tls connect error" in body or "timed out" in body:
            return True

    return False


def _normalize_fallback_image_url(url: str) -> str:
    """下载失败时的兜底 URL 规范化。"""
    raw = str(url or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if raw.startswith("/"):
        return f"https://assets.grok.com{raw}"
    return f"https://assets.grok.com/{raw}"


def _build_parent_source_candidates(parent_post_id: str, source_image_url: str) -> List[str]:
    """构建 parentPostId 编辑可复用的图片源候选列表。"""
    candidates: List[str] = []

    raw = str(source_image_url or "").strip()
    if raw:
        candidates.append(_normalize_fallback_image_url(raw))

    parent = str(parent_post_id or "").strip()
    if parent:
        candidates.extend(
            [
                f"https://imagine-public.x.ai/imagine-public/share-images/{parent}.png",
                f"https://imagine-public.x.ai/imagine-public/share-images/{parent}.jpg",
                f"https://imagine-public.x.ai/imagine-public/share-images/{parent}.jpeg",
                f"https://imagine-public.x.ai/imagine-public/images/{parent}.png",
                f"https://imagine-public.x.ai/imagine-public/images/{parent}.jpg",
                f"https://imagine-public.x.ai/imagine-public/images/{parent}.jpeg",
            ]
        )

    seen: set[str] = set()
    normalized: List[str] = []
    for item in candidates:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _should_skip_parent_precreate(image_ref: str) -> bool:
    """公开分享图直接交给 app-chat 预处理，不在本地显式 create post。"""
    raw = str(image_ref or "").strip().lower()
    if "imagine-public.x.ai/imagine-public/share-images/" in raw:
        return True
    return False


class ImageEditService:
    """Image edit orchestration service."""

    async def _emit_progress(
        self,
        progress_cb: Callable[[str, dict], Any] | None,
        event: str,
        progress: int,
        message: str,
        **extra: Any,
    ) -> None:
        if not progress_cb:
            return
        payload = {"progress": int(progress), "message": message}
        if extra:
            payload.update(extra)
        try:
            result = progress_cb(event, payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.debug(f"Image edit progress callback ignored: {e}")

    async def edit(
        self,
        *,
        token_mgr: Any,
        token: str,
        model_info: Any,
        prompt: str,
        images: List[str],
        n: int,
        response_format: str,
        stream: bool,
        return_all_images: bool = False,
        progress_cb: Callable[[str, dict], Any] | None = None,
    ) -> ImageEditResult:
        max_token_retries = int(get_config("retry.max_retry"))
        tried_tokens: set[str] = set()
        last_error: Exception | None = None

        for attempt in range(max_token_retries):
            preferred = token if attempt == 0 else None
            current_token = await pick_token(
                token_mgr, model_info.model_id, tried_tokens, preferred=preferred
            )
            if not current_token:
                if last_error:
                    raise last_error
                raise AppException(
                    message="No available tokens. Please try again later.",
                    error_type=ErrorType.RATE_LIMIT.value,
                    code="rate_limit_exceeded",
                    status_code=429,
                )

            tried_tokens.add(current_token)
            await self._emit_progress(
                progress_cb,
                "token_selected",
                8,
                "已匹配编辑令牌",
            )
            try:
                await self._emit_progress(
                    progress_cb,
                    "upload_start",
                    16,
                    "正在上传输入图片",
                )
                image_urls = await self._upload_images(images, current_token)
                await self._emit_progress(
                    progress_cb,
                    "upload_done",
                    30,
                    f"图片上传完成，共 {len(image_urls)} 张",
                    count=len(image_urls),
                )
                await self._emit_progress(
                    progress_cb,
                    "pre_create_start",
                    36,
                    "正在创建媒体帖子",
                )
                parent_post_id = await self._get_parent_post_id(
                    current_token, image_urls
                )
                await self._emit_progress(
                    progress_cb,
                    "pre_create_done",
                    42,
                    "媒体帖子创建完成",
                    parent_post_id=parent_post_id or "",
                )

                model_config_override = {
                    "modelMap": {
                        "imageEditModel": "imagine",
                        "imageEditModelConfig": {
                            "imageReferences": image_urls,
                        },
                    }
                }
                if parent_post_id:
                    model_config_override["modelMap"]["imageEditModelConfig"][
                        "parentPostId"
                    ] = parent_post_id

                tool_overrides = {"imageGen": True}

                if stream:
                    response = await GrokChatService().chat(
                        token=current_token,
                        message=prompt,
                        model=model_info.grok_model,
                        mode=None,
                        stream=True,
                        tool_overrides=tool_overrides,
                        model_config_override=model_config_override,
                        image_generation_count=1,
                    )
                    processor = ImageStreamProcessor(
                        model_info.model_id,
                        current_token,
                        n=n,
                        response_format=response_format,
                    )
                    return ImageEditResult(
                        stream=True,
                        data=wrap_stream_with_usage(
                            processor.process(response),
                            token_mgr,
                            current_token,
                            model_info.model_id,
                        ),
                        token_used=current_token,
                    )

                await self._emit_progress(
                    progress_cb,
                    "chat_request_start",
                    48,
                    "已提交编辑请求",
                    parent_post_id=parent_post_id or "",
                )
                images_out = await self._collect_images(
                    token=current_token,
                    prompt=prompt,
                    model_info=model_info,
                    response_format=response_format,
                    tool_overrides=tool_overrides,
                    model_config_override=model_config_override,
                    return_all_images=return_all_images,
                    progress_cb=progress_cb,
                )
                await self._emit_progress(
                    progress_cb,
                    "collect_done",
                    92,
                    f"已收到 {len(images_out)} 张结果",
                )
                try:
                    effort = (
                        EffortType.HIGH
                        if (model_info and model_info.cost.value == "high")
                        else EffortType.LOW
                    )
                    await token_mgr.consume(current_token, effort)
                    logger.debug(
                        f"Image edit completed, recorded usage (effort={effort.value})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to record image edit usage: {e}")
                return ImageEditResult(
                    stream=False,
                    data=images_out,
                    token_used=current_token,
                )

            except UpstreamException as e:
                last_error = e
                if rate_limited(e):
                    await token_mgr.mark_rate_limited(current_token)
                    await self._emit_progress(
                        progress_cb,
                        "rate_limited",
                        16,
                        "令牌限流，正在切换重试",
                    )
                    logger.warning(
                        f"Token {current_token[:10]}... rate limited (429), "
                        f"trying next token (attempt {attempt + 1}/{max_token_retries})"
                    )
                    continue
                raise

        if last_error:
            raise last_error
        raise AppException(
            message="No available tokens. Please try again later.",
            error_type=ErrorType.RATE_LIMIT.value,
            code="rate_limit_exceeded",
            status_code=429,
        )

    async def edit_with_parent_post(
        self,
        *,
        token_mgr: Any,
        token: str,
        model_info: Any,
        prompt: str,
        parent_post_id: str,
        source_image_url: str,
        response_format: str,
        stream: bool,
        return_all_images: bool = False,
        progress_cb: Callable[[str, dict], Any] | None = None,
    ) -> ImageEditResult:
        """基于 parentPostId 进行编辑，不上传图片。"""
        max_token_retries = int(get_config("retry.max_retry"))
        tried_tokens: set[str] = set()
        last_error: Exception | None = None

        for attempt in range(max_token_retries):
            preferred = token if attempt == 0 else None
            current_token = await pick_token(
                token_mgr, model_info.model_id, tried_tokens, preferred=preferred
            )
            if not current_token:
                if last_error:
                    raise last_error
                raise AppException(
                    message="No available tokens. Please try again later.",
                    error_type=ErrorType.RATE_LIMIT.value,
                    code="rate_limit_exceeded",
                    status_code=429,
                )

            tried_tokens.add(current_token)
            await self._emit_progress(
                progress_cb,
                "token_selected",
                8,
                "已匹配编辑令牌",
            )
            try:
                source_candidates = _build_parent_source_candidates(
                    parent_post_id, source_image_url
                )
                image_ref = source_candidates[0] if source_candidates else ""
                effective_parent_post_id = parent_post_id
                await self._emit_progress(
                    progress_cb,
                    "pre_create_start",
                    18,
                    "正在创建媒体帖子",
                    parent_post_id=parent_post_id,
                )
                if _should_skip_parent_precreate(image_ref):
                    logger.info(
                        "Image edit(parentPostId) skip pre-create for public share image: "
                        f"parent_post_id={parent_post_id}, media_url={image_ref}"
                    )
                    await self._emit_progress(
                        progress_cb,
                        "pre_create_skipped",
                        26,
                        "已识别公开分享图，跳过媒体帖子创建",
                    )
                else:
                    try:
                        # 同账号原图仍保留预创建逻辑，提升 assets/content 链路稳定性。
                        image_post_id = ""
                        precreate_errors: List[str] = []
                        for candidate in source_candidates:
                            try:
                                image_post_id = await VideoService().create_image_post(
                                    current_token, candidate
                                )
                                if image_post_id:
                                    image_ref = candidate
                                    effective_parent_post_id = image_post_id
                                    break
                            except Exception as candidate_error:
                                precreate_errors.append(str(candidate_error))
                        if not image_post_id and precreate_errors:
                            raise UpstreamException(precreate_errors[-1])
                        logger.info(
                            "Image edit(parentPostId) pre-create media post done: "
                            f"parent_post_id={parent_post_id}, "
                            f"image_post_id={effective_parent_post_id}, media_url={image_ref}"
                        )
                        await self._emit_progress(
                            progress_cb,
                            "pre_create_done",
                            34,
                            "媒体帖子创建完成",
                            image_post_id=effective_parent_post_id,
                        )
                    except Exception as e:
                        logger.warning(
                            "Image edit(parentPostId) pre-create media post failed, continue anyway: "
                            f"parent_post_id={parent_post_id}, media_url={image_ref}, error={e}"
                        )
                        await self._emit_progress(
                            progress_cb,
                            "pre_create_failed",
                            28,
                            "媒体帖子创建失败，继续请求",
                        )

                model_config_override = {
                    "modelMap": {
                        "imageEditModel": "imagine",
                        "imageEditModelConfig": {
                            "imageReferences": [image_ref],
                            "parentPostId": effective_parent_post_id,
                        },
                    }
                }
                tool_overrides = {"imageGen": True}

                if stream:
                    response = await GrokChatService().chat(
                        token=current_token,
                        message=prompt,
                        model=model_info.grok_model,
                        mode=None,
                        stream=True,
                        tool_overrides=tool_overrides,
                        model_config_override=model_config_override,
                        image_generation_count=1,
                    )
                    processor = ImageStreamProcessor(
                        model_info.model_id,
                        current_token,
                        n=1,
                        response_format=response_format,
                    )
                    return ImageEditResult(
                        stream=True,
                        data=wrap_stream_with_usage(
                            processor.process(response),
                            token_mgr,
                            current_token,
                            model_info.model_id,
                        ),
                        token_used=current_token,
                    )

                await self._emit_progress(
                    progress_cb,
                    "chat_request_start",
                    48,
                    "已提交编辑请求",
                    parent_post_id=effective_parent_post_id,
                )
                images_out = await self._collect_images(
                    token=current_token,
                    prompt=prompt,
                    model_info=model_info,
                    response_format=response_format,
                    tool_overrides=tool_overrides,
                    model_config_override=model_config_override,
                    return_all_images=return_all_images,
                    progress_cb=progress_cb,
                )
                await self._emit_progress(
                    progress_cb,
                    "collect_done",
                    92,
                    f"已收到 {len(images_out)} 张结果",
                )
                try:
                    effort = (
                        EffortType.HIGH
                        if (model_info and model_info.cost.value == "high")
                        else EffortType.LOW
                    )
                    await token_mgr.consume(current_token, effort)
                    logger.debug(
                        "Image edit(parentPostId) completed, "
                        f"recorded usage (effort={effort.value})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to record image edit(parentPostId) usage: {e}")
                return ImageEditResult(
                    stream=False,
                    data=images_out,
                    token_used=current_token,
                )

            except UpstreamException as e:
                last_error = e
                if rate_limited(e):
                    await token_mgr.mark_rate_limited(current_token)
                    await self._emit_progress(
                        progress_cb,
                        "rate_limited",
                        16,
                        "令牌限流，正在切换重试",
                    )
                    logger.warning(
                        f"Token {current_token[:10]}... rate limited (429), "
                        f"trying next token (attempt {attempt + 1}/{max_token_retries})"
                    )
                    continue
                raise

        if last_error:
            raise last_error
        raise AppException(
            message="No available tokens. Please try again later.",
            error_type=ErrorType.RATE_LIMIT.value,
            code="rate_limit_exceeded",
            status_code=429,
        )

    async def _upload_images(self, images: List[str], token: str) -> List[str]:
        image_urls: List[str] = []
        upload_service = UploadService()
        try:
            for image in images:
                _, file_uri = await upload_service.upload_file(image, token)
                if file_uri:
                    if file_uri.startswith("http"):
                        image_urls.append(file_uri)
                    else:
                        image_urls.append(
                            f"https://assets.grok.com/{file_uri.lstrip('/')}"
                        )
        except Exception as e:
            if _is_upload_rejected_error(e):
                raise AppException(
                    message="图片上传被拒绝，请更换图片后重试",
                    error_type=ErrorType.INVALID_REQUEST.value,
                    code="upload_rejected",
                    status_code=400,
                )
            if _is_upload_network_error(e):
                raise AppException(
                    message="图片上传失败：网络连接异常，请稍后重试",
                    error_type=ErrorType.SERVER.value,
                    code="upload_network_error",
                    status_code=502,
                )
            raise AppException(
                message="图片上传失败，请稍后重试",
                error_type=ErrorType.SERVER.value,
                code="upload_failed",
                status_code=502,
            )
        finally:
            await upload_service.close()

        if not image_urls:
            raise AppException(
                message="Image upload failed",
                error_type=ErrorType.SERVER.value,
                code="upload_failed",
            )

        return image_urls

    async def _get_parent_post_id(self, token: str, image_urls: List[str]) -> str:
        parent_post_id = None
        try:
            media_service = VideoService()
            parent_post_id = await media_service.create_image_post(token, image_urls[0])
            logger.debug(f"Parent post ID: {parent_post_id}")
        except Exception as e:
            logger.warning(f"Create image post failed: {e}")

        if parent_post_id:
            return parent_post_id

        for url in image_urls:
            match = re.search(r"/generated/([a-f0-9-]+)/", url)
            if match:
                parent_post_id = match.group(1)
                logger.debug(f"Parent post ID: {parent_post_id}")
                break
            match = re.search(r"/users/[^/]+/([a-f0-9-]+)/content", url)
            if match:
                parent_post_id = match.group(1)
                logger.debug(f"Parent post ID: {parent_post_id}")
                break

        return parent_post_id or ""

    async def _collect_images(
        self,
        *,
        token: str,
        prompt: str,
        model_info: Any,
        response_format: str,
        tool_overrides: dict,
        model_config_override: dict,
        return_all_images: bool = False,
        progress_cb: Callable[[str, dict], Any] | None = None,
    ) -> List[str]:
        async def _call_edit():
            response = await GrokChatService().chat(
                token=token,
                message=prompt,
                model=model_info.grok_model,
                mode=None,
                stream=True,
                tool_overrides=tool_overrides,
                model_config_override=model_config_override,
                image_generation_count=1,
            )
            processor = ImageCollectProcessor(
                model_info.model_id,
                token,
                response_format=response_format,
                progress_cb=progress_cb,
            )
            return await processor.process(response)

        all_images = await _call_edit()

        if not all_images:
            raise UpstreamException(
                "Image edit returned no results", details={"error": "empty_result"}
            )
        share_items = []
        if token:
            for image_url in all_images:
                post_id = _extract_image_post_id(image_url)
                if post_id and all(post_id != exist_id for exist_id, _ in share_items):
                    share_items.append((post_id, image_url))
            for post_id, image_url in share_items:
                await _try_log_image_share_link(token, post_id, local_url=image_url)
        if return_all_images:
            return all_images
        return [all_images[0]]


class ImageStreamProcessor(BaseProcessor):
    """HTTP image stream processor."""

    def __init__(
        self, model: str, token: str = "", n: int = 1, response_format: str = "b64_json"
    ):
        super().__init__(model, token)
        self.partial_index = 0
        self.n = n
        self.target_index = 0 if n == 1 else None
        self.response_format = response_format
        if response_format == "url":
            self.response_field = "url"
        elif response_format == "base64":
            self.response_field = "base64"
        else:
            self.response_field = "b64_json"

    def _sse(self, event: str, data: dict) -> str:
        """Build SSE response."""
        return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"

    async def process(
        self, response: AsyncIterable[bytes]
    ) -> AsyncGenerator[str, None]:
        """Process stream response."""
        final_images = []
        idle_timeout = get_config("image.stream_timeout")

        try:
            async for line in _with_idle_timeout(response, idle_timeout, self.model):
                line = _normalize_line(line)
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                resp = data.get("result", {}).get("response", {})

                # Image generation progress
                if img := resp.get("streamingImageGenerationResponse"):
                    image_index = img.get("imageIndex", 0)
                    progress = img.get("progress", 0)

                    if self.n == 1 and image_index != self.target_index:
                        continue

                    out_index = 0 if self.n == 1 else image_index

                    yield self._sse(
                        "image_generation.partial_image",
                        {
                            "type": "image_generation.partial_image",
                            self.response_field: "",
                            "index": out_index,
                            "progress": progress,
                        },
                    )
                    continue

                # modelResponse
                if mr := resp.get("modelResponse"):
                    if urls := _collect_images(mr):
                        for url in urls:
                            if self.response_format == "url":
                                try:
                                    processed = await self.process_url(url, "image")
                                except Exception as e:
                                    logger.warning(
                                        "Image stream URL resolve failed, fallback to raw URL: "
                                        f"error={e}"
                                    )
                                    processed = _normalize_fallback_image_url(url)
                                if processed:
                                    final_images.append(processed)
                                continue
                            try:
                                dl_service = self._get_dl()
                                base64_data = await dl_service.parse_b64(
                                    url, self.token, "image"
                                )
                                if base64_data:
                                    if "," in base64_data:
                                        b64 = base64_data.split(",", 1)[1]
                                    else:
                                        b64 = base64_data
                                    final_images.append(b64)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to convert image to base64, falling back to URL: {e}"
                                )
                                processed = await self.process_url(url, "image")
                                if processed:
                                    final_images.append(processed)
                    continue

            for index, b64 in enumerate(final_images):
                if self.n == 1:
                    if index != self.target_index:
                        continue
                    out_index = 0
                else:
                    out_index = index

                yield self._sse(
                    "image_generation.completed",
                    {
                        "type": "image_generation.completed",
                        self.response_field: b64,
                        "index": out_index,
                        "usage": {
                            "total_tokens": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "input_tokens_details": {
                                "text_tokens": 0,
                                "image_tokens": 0,
                            },
                        },
                    },
                )
                post_id = _extract_image_post_id(b64)
                if post_id and self.token:
                    await _try_log_image_share_link(self.token, post_id, local_url=b64)
        except asyncio.CancelledError:
            logger.debug("Image stream cancelled by client")
        except StreamIdleTimeoutError as e:
            raise UpstreamException(
                message=f"Image stream idle timeout after {e.idle_seconds}s",
                status_code=504,
                details={
                    "error": str(e),
                    "type": "stream_idle_timeout",
                    "idle_seconds": e.idle_seconds,
                },
            )
        except RequestsError as e:
            if _is_http2_error(e):
                logger.warning(f"HTTP/2 stream error in image: {e}")
                raise UpstreamException(
                    message="Upstream connection closed unexpectedly",
                    status_code=502,
                    details={"error": str(e), "type": "http2_stream_error"},
                )
            logger.error(f"Image stream request error: {e}")
            raise UpstreamException(
                message=f"Upstream request failed: {e}",
                status_code=502,
                details={"error": str(e)},
            )
        except Exception as e:
            logger.error(
                f"Image stream processing error: {e}",
                extra={"error_type": type(e).__name__},
            )
            raise
        finally:
            await self.close()


class ImageCollectProcessor(BaseProcessor):
    """HTTP image non-stream processor."""

    def __init__(
        self,
        model: str,
        token: str = "",
        response_format: str = "b64_json",
        progress_cb: Callable[[str, dict], Any] | None = None,
    ):
        if response_format == "base64":
            response_format = "b64_json"
        super().__init__(model, token)
        self.response_format = response_format
        self.progress_cb = progress_cb

    async def _emit_progress(
        self, event: str, progress: int, message: str, **extra: Any
    ) -> None:
        if not self.progress_cb:
            return
        payload = {"progress": int(progress), "message": message}
        if extra:
            payload.update(extra)
        try:
            result = self.progress_cb(event, payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    async def process(self, response: AsyncIterable[bytes]) -> List[str]:
        """Process and collect images."""
        images = []
        idle_timeout = get_config("image.stream_timeout")
        chat_connected_emitted = False

        try:
            async for line in _with_idle_timeout(response, idle_timeout, self.model):
                line = _normalize_line(line)
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                resp = data.get("result", {}).get("response", {})
                if not chat_connected_emitted and resp:
                    chat_connected_emitted = True
                    await self._emit_progress(
                        "chat_connected",
                        60,
                        "模型连接成功，正在生成图片",
                    )

                if mr := resp.get("modelResponse"):
                    if urls := _collect_images(mr):
                        for url in urls:
                            if self.response_format == "url":
                                try:
                                    processed = await self.process_url(url, "image")
                                except Exception as e:
                                    logger.warning(
                                        "Image collect URL resolve failed, fallback to raw URL: "
                                        f"error={e}"
                                    )
                                    processed = _normalize_fallback_image_url(url)
                                if processed:
                                    images.append(processed)
                                    progress = min(90, 64 + len(images) * 12)
                                    await self._emit_progress(
                                        "image_downloaded",
                                        progress,
                                        f"已下载第 {len(images)} 张图片",
                                        count=len(images),
                                    )
                                continue
                            try:
                                dl_service = self._get_dl()
                                base64_data = await dl_service.parse_b64(
                                    url, self.token, "image"
                                )
                                if base64_data:
                                    if "," in base64_data:
                                        b64 = base64_data.split(",", 1)[1]
                                    else:
                                        b64 = base64_data
                                    images.append(b64)
                                    progress = min(90, 64 + len(images) * 12)
                                    await self._emit_progress(
                                        "image_downloaded",
                                        progress,
                                        f"已下载第 {len(images)} 张图片",
                                        count=len(images),
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to convert image to base64, falling back to URL: {e}"
                                )
                                processed = await self.process_url(url, "image")
                                if processed:
                                    images.append(processed)
                                    progress = min(90, 64 + len(images) * 12)
                                    await self._emit_progress(
                                        "image_downloaded",
                                        progress,
                                        f"已下载第 {len(images)} 张图片",
                                        count=len(images),
                                    )

        except asyncio.CancelledError:
            logger.debug("Image collect cancelled by client")
        except StreamIdleTimeoutError as e:
            logger.warning(f"Image collect idle timeout: {e}")
        except RequestsError as e:
            if _is_http2_error(e):
                logger.warning(f"HTTP/2 stream error in image collect: {e}")
            else:
                logger.error(f"Image collect request error: {e}")
        except Exception as e:
            logger.error(
                f"Image collect processing error: {e}",
                extra={"error_type": type(e).__name__},
            )
        finally:
            await self.close()

        return images


__all__ = ["ImageEditService", "ImageEditResult"]
