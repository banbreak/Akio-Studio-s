"""Metrics ingestion and greenlight-validation engine.

Implements Pillar 1 of Akio Studio: pull short-form performance metrics,
compute the retention-to-engagement composite, and decide whether a concept
has earned a 16:9 episode greenlight.

Audit findings honored here:

* **B7** — :class:`PostMetrics` carries ``watch_time_50pct_count`` (the
  midpoint-retention numerator the spec forgot) and ``posted_at`` so the
  "3 consecutive uploads" rule is defined chronologically.
* **M6** — denominators never mix: the retention family (R3s, R50, CR) is
  per-*view*; engagement velocity is per-*impression*. Zero-view or
  zero-impression posts are reported as *not evaluable* instead of raising
  ``ZeroDivisionError`` or silently counting as failures.
* **M4** — audience lore-theory mining is pure-stdlib lexical clustering; no
  sentence-transformers embedding model is ever loaded.
* **A6** — the greenlight webhook payload is shaped per host (Discord wants
  ``{"content": ...}``, Slack wants ``{"text": ...}``), 204 counts as
  success, and ``Retry-After`` is honored on 429.

Thresholds always come from :class:`akio_studio.config.GreenlightThresholds`;
nothing metric-related is hard-coded here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from akio_studio.config import StudioConfig
from akio_studio.exceptions import MetricsError, WebhookDispatchError

logger = logging.getLogger(__name__)

#: Per-attempt webhook POST timeout, seconds.
_WEBHOOK_TIMEOUT_S: float = 10.0

#: Exponential backoff delays between webhook attempts, seconds.
_WEBHOOK_BACKOFF_S: tuple[float, ...] = (1.0, 2.0, 4.0)

#: Maximum delay honored from a 429 ``Retry-After`` header, seconds.
_RETRY_AFTER_CAP_S: float = 30.0

#: Minimum Jaccard similarity for a comment to join an existing theory cluster.
_CLUSTER_SIMILARITY_THRESHOLD: float = 0.35

#: Fixed epoch for deterministic mock ``posted_at`` derivation.
_MOCK_EPOCH: datetime = datetime(2025, 6, 1, tzinfo=UTC)

#: Deterministic pool of plausible fan comments for the mock fetcher.
_MOCK_COMMENT_POOL: tuple[str, ...] = (
    "the masked mentor is definitely aki's father, rewatch the shrine scene",
    "hear me out: the shrine fox is the same spirit from the first short",
    "that sword glow matches the festival lanterns, foreshadowing confirmed",
    "nobody noticed the mentor never steps on sacred ground?? he's a spirit",
    "the fox spirit theory keeps getting stronger every upload",
    "animation on the rain scene was insane, studio quality",
    "aki's pendant is the sealed half of the shrine relic, calling it now",
    "the lantern festival is a memorial for the war the mentor started",
    "ok but why does the fox have the same scar as the mentor",
    "this is my favorite series on this app rn",
)

#: Small built-in English stopword set for comment normalization (audit M4).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at",
        "to", "for", "with", "is", "are", "was", "were", "be", "been",
        "being", "it", "its", "this", "that", "these", "those", "i", "you",
        "he", "she", "we", "they", "them", "his", "her", "my", "your", "our",
        "their", "me", "us", "so", "just", "not", "no", "do", "does", "did",
        "have", "has", "had", "what", "when", "where", "who", "why", "how",
        "will", "would", "can", "could", "should", "than", "then", "there",
        "here", "from", "by", "as", "about", "into", "over", "after",
        "before", "all", "any", "some", "out", "up", "down", "very", "too",
        "also", "really", "rn", "ok", "im", "u", "lol", "omg", "tbh", "btw",
    }
)

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(comment: str) -> frozenset[str]:
    """Normalize a comment to its set of content tokens.

    Casefolds, strips URLs and @-mentions, tokenizes on alphanumerics, and
    drops stopwords and single-character tokens.
    """
    text = _MENTION_RE.sub(" ", _URL_RE.sub(" ", comment.casefold()))
    return frozenset(
        token
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token not in _STOPWORDS
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets; 0.0 when both are empty."""
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _centroid(members: list[tuple[int, str, frozenset[str]]]) -> frozenset[str]:
    """Majority-token centroid: tokens present in at least half the members."""
    counts: Counter[str] = Counter()
    for _, _, tokens in members:
        counts.update(tokens)
    size = len(members)
    return frozenset(token for token, count in counts.items() if count * 2 >= size)


@dataclass(frozen=True)
class PostMetrics:
    """Immutable metrics snapshot for one published short-form post.

    Retention counters form a monotonically non-increasing funnel per view:
    ``views >= watch_time_3s_count >= watch_time_50pct_count >=
    completion_count``. ``watch_time_50pct_count`` is the audit-added (B7)
    midpoint-retention numerator; ``posted_at`` orders uploads for the
    consecutive-greenlight rule and must be timezone-aware.
    """

    post_id: str
    concept_id: str
    posted_at: datetime
    views: int
    impressions: int
    watch_time_3s_count: int
    watch_time_50pct_count: int
    completion_count: int
    likes: int
    comments_text_list: list[str]
    shares: int
    saves: int

    def __post_init__(self) -> None:
        """Validate counter consistency; raise :class:`MetricsError` on violation."""
        for name in (
            "views",
            "impressions",
            "watch_time_3s_count",
            "watch_time_50pct_count",
            "completion_count",
            "likes",
            "shares",
            "saves",
        ):
            value: int = getattr(self, name)
            if value < 0:
                raise MetricsError(
                    f"post {self.post_id!r}: {name} must be non-negative, got {value}"
                )
        if self.views > self.impressions:
            raise MetricsError(
                f"post {self.post_id!r}: views ({self.views}) cannot exceed "
                f"impressions ({self.impressions})"
            )
        for name in ("watch_time_3s_count", "watch_time_50pct_count", "completion_count"):
            value = getattr(self, name)
            if value > self.views:
                raise MetricsError(
                    f"post {self.post_id!r}: {name} ({value}) cannot exceed "
                    f"views ({self.views})"
                )
        if self.watch_time_50pct_count > self.watch_time_3s_count:
            raise MetricsError(
                f"post {self.post_id!r}: watch_time_50pct_count "
                f"({self.watch_time_50pct_count}) cannot exceed watch_time_3s_count "
                f"({self.watch_time_3s_count}); retention is non-increasing"
            )
        if self.completion_count > self.watch_time_50pct_count:
            raise MetricsError(
                f"post {self.post_id!r}: completion_count ({self.completion_count}) "
                f"cannot exceed watch_time_50pct_count "
                f"({self.watch_time_50pct_count}); retention is non-increasing"
            )
        if self.posted_at.tzinfo is None:
            raise MetricsError(
                f"post {self.post_id!r}: posted_at must be timezone-aware "
                "(naive datetimes cannot be ordered against aware ones)"
            )


class MetricsIngestionEngine:
    """Ingests platform metrics and applies the greenlight validation rules.

    All thresholds come from ``config.thresholds``
    (:class:`akio_studio.config.GreenlightThresholds`); the engine itself
    holds no magic numbers for gates.
    """

    def __init__(self, config: StudioConfig | None = None) -> None:
        """Create the engine with an optional injected :class:`StudioConfig`."""
        self._config = config if config is not None else StudioConfig()

    def fetch_mock_platform_data(
        self, post_id: str, concept_id: str = "concept-unknown"
    ) -> PostMetrics:
        """Return deterministic, plausible mock metrics for ``post_id``.

        The RNG is seeded with the ``post_id`` string and ``posted_at`` is
        derived from a fixed epoch plus a SHA-256-based offset, so identical
        inputs always yield identical metrics (stable for tests — Python's
        built-in ``hash`` is process-randomized and is deliberately avoided).
        Each retention counter is derived as a fraction of the previous one,
        so the monotonic funnel invariants hold by construction.
        """
        rng = random.Random(post_id)
        impressions = rng.randint(2_000, 250_000)
        views = int(impressions * rng.uniform(0.20, 0.70))
        watch_3s = int(views * rng.uniform(0.50, 0.95))
        watch_50 = int(watch_3s * rng.uniform(0.40, 0.90))
        completions = int(watch_50 * rng.uniform(0.40, 0.95))
        likes = int(views * rng.uniform(0.010, 0.150))
        shares = int(views * rng.uniform(0.001, 0.030))
        saves = int(views * rng.uniform(0.001, 0.040))
        comments = [
            rng.choice(_MOCK_COMMENT_POOL) for _ in range(rng.randint(4, 12))
        ]
        digest = hashlib.sha256(post_id.encode("utf-8")).digest()
        offset_s = int.from_bytes(digest[:8], "big") % (90 * 24 * 3600)
        posted_at = _MOCK_EPOCH + timedelta(seconds=offset_s)
        metrics = PostMetrics(
            post_id=post_id,
            concept_id=concept_id,
            posted_at=posted_at,
            views=views,
            impressions=impressions,
            watch_time_3s_count=watch_3s,
            watch_time_50pct_count=watch_50,
            completion_count=completions,
            likes=likes,
            comments_text_list=comments,
            shares=shares,
            saves=saves,
        )
        logger.debug("mock metrics for %s: views=%d impressions=%d", post_id, views, impressions)
        return metrics

    def calculate_retention_to_engagement(self, metrics: PostMetrics) -> dict[str, object]:
        """Compute the retention-to-engagement report for one post.

        Denominators (audit M6, never mixed):

        * ``r_3s = watch_time_3s_count / views`` — per view
        * ``r_50 = watch_time_50pct_count / views`` — per view
        * ``cr = completion_count / views`` — per view
        * ``engagement_velocity_pct = ((shares*3.0 + saves*2.0 +
          comments*1.5 + likes*0.5) / impressions) * 100`` — per impression,
          where ``comments = len(comments_text_list)``

        ``r2e_score`` is a composite in ``[0, 1]``: the arithmetic mean of
        each of the four metrics normalized by its configured threshold and
        capped at 1.0 —
        ``mean(min(r_3s/T_3s, 1), min(r_50/T_50, 1), min(cr/T_cr, 1),
        min(ev/T_ev, 1))`` — so 1.0 means every gate is met or exceeded.

        Zero-view or zero-impression posts return ``evaluable=False`` with
        ``None`` metric values and ``passes=False`` (never a
        ``ZeroDivisionError``); their threshold checks are all ``False``.
        """
        thresholds = self._config.thresholds
        check_names = (
            "hook_retention_3s",
            "midpoint_retention",
            "completion_rate",
            "engagement_velocity_pct",
        )
        if metrics.views == 0 or metrics.impressions == 0:
            logger.info(
                "post %s not evaluable (views=%d, impressions=%d)",
                metrics.post_id,
                metrics.views,
                metrics.impressions,
            )
            return {
                "post_id": metrics.post_id,
                "evaluable": False,
                "r_3s": None,
                "r_50": None,
                "cr": None,
                "engagement_velocity_pct": None,
                "r2e_score": None,
                "passes": False,
                "checks": {name: False for name in check_names},
            }

        r_3s = metrics.watch_time_3s_count / metrics.views
        r_50 = metrics.watch_time_50pct_count / metrics.views
        cr = metrics.completion_count / metrics.views
        weighted_engagement = (
            metrics.shares * 3.0
            + metrics.saves * 2.0
            + len(metrics.comments_text_list) * 1.5
            + metrics.likes * 0.5
        )
        ev = (weighted_engagement / metrics.impressions) * 100.0

        checks = {
            "hook_retention_3s": r_3s >= thresholds.hook_retention_3s,
            "midpoint_retention": r_50 >= thresholds.midpoint_retention,
            "completion_rate": cr >= thresholds.completion_rate,
            "engagement_velocity_pct": ev >= thresholds.engagement_velocity_pct,
        }

        def norm(value: float, threshold: float) -> float:
            return 1.0 if threshold <= 0 else min(value / threshold, 1.0)

        r2e_score = (
            norm(r_3s, thresholds.hook_retention_3s)
            + norm(r_50, thresholds.midpoint_retention)
            + norm(cr, thresholds.completion_rate)
            + norm(ev, thresholds.engagement_velocity_pct)
        ) / 4.0

        return {
            "post_id": metrics.post_id,
            "evaluable": True,
            "r_3s": r_3s,
            "r_50": r_50,
            "cr": cr,
            "engagement_velocity_pct": ev,
            "r2e_score": r2e_score,
            "passes": all(checks.values()),
            "checks": checks,
        }

    def evaluate_concept_validation(self, concept_id: str, posts: list[PostMetrics]) -> bool:
        """Decide whether ``concept_id`` has earned an episode greenlight.

        Filters ``posts`` to the concept, sorts by ``posted_at`` ascending
        (the authoritative order — caller order is never trusted), and
        requires the **most recent**
        ``config.thresholds.consecutive_uploads_required`` uploads to each be
        evaluable and pass all four thresholds (audit B7: "3 consecutive"
        means the last 3 chronologically — an any-3 window would greenlight
        a concept whose latest upload flopped).
        """
        required = self._config.thresholds.consecutive_uploads_required
        # post_id tie-break: identical posted_at values must not make the
        # window boundary depend on caller list order.
        matching = sorted(
            (post for post in posts if post.concept_id == concept_id),
            key=lambda post: (post.posted_at, post.post_id),
        )
        if len(matching) < required:
            logger.info(
                "concept %s: %d matching post(s), %d required — no greenlight",
                concept_id,
                len(matching),
                required,
            )
            return False
        recent = matching[-required:]
        for post in recent:
            report = self.calculate_retention_to_engagement(post)
            if not (report["evaluable"] and report["passes"]):
                logger.info(
                    "concept %s: post %s fails greenlight gate (evaluable=%s, checks=%s)",
                    concept_id,
                    post.post_id,
                    report["evaluable"],
                    report["checks"],
                )
                return False
        logger.info(
            "concept %s: last %d uploads all pass — greenlight", concept_id, required
        )
        return True

    def mine_audience_lore_theories(
        self, comments_list: list[str], top_n: int = 5
    ) -> list[str]:
        """Extract up to ``top_n`` representative fan-theory comments.

        Pure-stdlib lexical clustering (audit M4 — no embedding model):
        comments are normalized (casefolded, URLs/@-mentions stripped,
        tokenized, stopwords dropped) and comments with fewer than two
        content tokens are skipped. Each comment greedily joins the existing
        cluster whose majority-token centroid it is most Jaccard-similar to
        (Jaccard >= ``_CLUSTER_SIMILARITY_THRESHOLD``, 0.35), else it starts
        a new cluster. Clusters are
        ranked by size (ties: earliest-formed first) and each contributes its
        medoid — the member with the highest mean token-set similarity to its
        cluster-mates. Deterministic for identical input.
        """
        if top_n <= 0:
            return []
        prepared: list[tuple[int, str, frozenset[str]]] = []
        for index, raw in enumerate(comments_list):
            tokens = _content_tokens(raw)
            if len(tokens) < 2:
                continue
            prepared.append((index, raw, tokens))

        clusters: list[list[tuple[int, str, frozenset[str]]]] = []
        centroids: list[frozenset[str]] = []
        for entry in prepared:
            best_index = -1
            best_similarity = 0.0
            for cluster_index, centroid in enumerate(centroids):
                similarity = _jaccard(entry[2], centroid)
                if similarity > best_similarity:  # strict '>' keeps earliest on ties
                    best_similarity = similarity
                    best_index = cluster_index
            if best_index >= 0 and best_similarity >= _CLUSTER_SIMILARITY_THRESHOLD:
                clusters[best_index].append(entry)
                centroids[best_index] = _centroid(clusters[best_index])
            else:
                clusters.append([entry])
                centroids.append(entry[2])

        ranked = sorted(
            range(len(clusters)),
            key=lambda i: (-len(clusters[i]), clusters[i][0][0]),
        )
        theories: list[str] = []
        for cluster_index in ranked[:top_n]:
            members = clusters[cluster_index]
            if len(members) == 1:
                theories.append(members[0][1])
                continue
            best_comment = members[0][1]
            best_mean = -1.0
            for _, raw, tokens in members:
                total = sum(
                    _jaccard(tokens, other_tokens)
                    for _, _, other_tokens in members
                    if other_tokens is not tokens
                )
                mean_similarity = total / (len(members) - 1)
                if mean_similarity > best_mean:  # strict '>' keeps earliest on ties
                    best_mean = mean_similarity
                    best_comment = raw
            theories.append(best_comment)
        logger.debug(
            "mined %d theory cluster(s) from %d comment(s)", len(clusters), len(comments_list)
        )
        return theories

    async def dispatch_greenlight_webhook(
        self, concept_id: str, summary: dict[str, object], webhook_url: str
    ) -> None:
        """POST a greenlight notification, shaping the payload per host.

        Audit A6: Discord hosts get ``{"content": ..., "embeds": [...]}``,
        ``hooks.slack.com`` gets ``{"text": ...}``, any other host gets the
        raw ``{"concept_id": ..., "summary": ...}`` JSON. Only ``https://``
        URLs are accepted. The POST runs via ``urllib.request`` inside
        ``asyncio.to_thread`` with a 10 s per-attempt timeout; success is any
        HTTP 2xx (Discord returns 204). Failures are retried up to 3 times
        with exponential backoff (1 s, 2 s, 4 s); a 429 honors the
        ``Retry-After`` header when present (capped at 30 s). Raises
        :class:`WebhookDispatchError` on a non-https URL or after the final
        failed attempt.
        """
        parts = urllib.parse.urlsplit(webhook_url)
        if parts.scheme != "https" or not parts.hostname:
            raise WebhookDispatchError(
                f"webhook URL must be https:// with a host, got {webhook_url!r}"
            )
        payload = self._build_webhook_payload(concept_id, summary, parts.hostname)

        last_error = "no attempt made"
        total_attempts = len(_WEBHOOK_BACKOFF_S) + 1
        for attempt in range(1, total_attempts + 1):
            status: int | None = None
            retry_after: float | None = None
            try:
                status, retry_after = await asyncio.to_thread(
                    self._post_json, webhook_url, payload
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"network error: {exc}"
                logger.warning(
                    "greenlight webhook attempt %d/%d failed: %s",
                    attempt,
                    total_attempts,
                    last_error,
                )
            else:
                if 200 <= status < 300:
                    logger.info(
                        "greenlight webhook for %s delivered (HTTP %d, attempt %d)",
                        concept_id,
                        status,
                        attempt,
                    )
                    return
                last_error = f"HTTP {status}"
                logger.warning(
                    "greenlight webhook attempt %d/%d rejected: %s",
                    attempt,
                    total_attempts,
                    last_error,
                )
            if attempt <= len(_WEBHOOK_BACKOFF_S):
                delay = _WEBHOOK_BACKOFF_S[attempt - 1]
                if status == 429 and retry_after is not None:
                    delay = min(retry_after, _RETRY_AFTER_CAP_S)
                await asyncio.sleep(delay)
        raise WebhookDispatchError(
            f"greenlight webhook for {concept_id!r} failed after "
            f"{total_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _build_webhook_payload(
        concept_id: str, summary: dict[str, object], hostname: str
    ) -> dict[str, object]:
        """Shape the webhook JSON body for the destination host (audit A6)."""
        host = hostname.casefold()
        message = f"GREENLIGHT: concept '{concept_id}' cleared all metric gates."
        if (
            host in ("discord.com", "discordapp.com")
            or host.endswith(".discord.com")
            or host.endswith(".discordapp.com")
        ):
            fields = [
                {"name": str(key)[:256], "value": str(value)[:1024], "inline": True}
                for key, value in list(summary.items())[:10]
            ]
            return {
                "content": message,
                "embeds": [{"title": f"Greenlight: {concept_id}"[:256], "fields": fields}],
            }
        if host == "hooks.slack.com":
            lines = [message]
            lines.extend(f"{key}: {value}" for key, value in summary.items())
            return {"text": "\n".join(lines)}
        return {"concept_id": concept_id, "summary": summary}

    @staticmethod
    def _post_json(url: str, payload: dict[str, object]) -> tuple[int, float | None]:
        """Synchronous JSON POST; returns ``(status, retry_after_seconds)``.

        Runs inside ``asyncio.to_thread`` from the async dispatcher. HTTP
        error statuses are returned (with any parsed ``Retry-After``), not
        raised; transport failures propagate as ``URLError``/``OSError``.
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "akio-studio-metrics/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_WEBHOOK_TIMEOUT_S) as response:
                return int(response.status), None
        except urllib.error.HTTPError as exc:
            retry_after: float | None = None
            raw = exc.headers.get("Retry-After") if exc.headers else None
            if raw is not None:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
            return int(exc.code), retry_after
