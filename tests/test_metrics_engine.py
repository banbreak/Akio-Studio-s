"""Unit tests for :mod:`akio_studio.metrics_engine` — no network involved."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from akio_studio.exceptions import MetricsError, WebhookDispatchError
from akio_studio.metrics_engine import MetricsIngestionEngine, PostMetrics

T0 = datetime(2025, 6, 1, tzinfo=UTC)


def _post(
    post_id: str,
    posted_at: datetime,
    *,
    concept_id: str = "c1",
    views: int = 1000,
    impressions: int = 1000,
    w3: int = 800,
    w50: int = 500,
    comp: int = 400,
    likes: int = 30,
    comments: int = 4,
    shares: int = 20,
    saves: int = 10,
) -> PostMetrics:
    """Build a post; defaults pass every greenlight threshold."""
    return PostMetrics(
        post_id=post_id,
        concept_id=concept_id,
        posted_at=posted_at,
        views=views,
        impressions=impressions,
        watch_time_3s_count=w3,
        watch_time_50pct_count=w50,
        completion_count=comp,
        likes=likes,
        comments_text_list=["c"] * comments,
        shares=shares,
        saves=saves,
    )


def _failing_post(post_id: str, posted_at: datetime) -> PostMetrics:
    """A post whose 3-second hook retention (0.3) fails the 0.70 gate."""
    return _post(post_id, posted_at, w3=300, w50=250, comp=200)


class TestPostMetricsInvariants:
    """Constructor validation (audit M6/B7)."""

    def test_views_exceeding_impressions_raises(self) -> None:
        with pytest.raises(MetricsError):
            _post("p", T0, views=2000, impressions=1000)

    def test_midpoint_exceeding_hook_count_raises(self) -> None:
        with pytest.raises(MetricsError):
            _post("p", T0, w3=100, w50=200, comp=50)

    def test_completion_exceeding_midpoint_raises(self) -> None:
        with pytest.raises(MetricsError):
            _post("p", T0, w50=200, comp=300)

    def test_negative_counter_raises(self) -> None:
        with pytest.raises(MetricsError):
            _post("p", T0, likes=-1)

    def test_naive_posted_at_raises(self) -> None:
        with pytest.raises(MetricsError):
            _post("p", datetime(2025, 6, 1))  # noqa: DTZ001 — naive on purpose


class TestRetentionToEngagement:
    """Denominator separation and zero-view behavior (audit M6)."""

    def test_denominators_never_mix(self) -> None:
        engine = MetricsIngestionEngine()
        post = _post(
            "p",
            T0,
            views=500,
            impressions=2000,
            w3=400,
            w50=250,
            comp=100,
            likes=8,
            comments=4,
            shares=10,
            saves=5,
        )
        report = engine.calculate_retention_to_engagement(post)
        assert report["evaluable"] is True
        # Retention family is per-view.
        assert report["r_3s"] == pytest.approx(400 / 500)
        assert report["r_50"] == pytest.approx(250 / 500)
        assert report["cr"] == pytest.approx(100 / 500)
        # Engagement velocity is per-impression.
        expected_ev = (10 * 3.0 + 5 * 2.0 + 4 * 1.5 + 8 * 0.5) / 2000 * 100.0
        assert report["engagement_velocity_pct"] == pytest.approx(expected_ev)
        expected_r2e = (1.0 + 1.0 + (0.2 / 0.35) + (expected_ev / 8.5)) / 4.0
        assert report["r2e_score"] == pytest.approx(expected_r2e)
        assert report["passes"] is False

    def test_zero_views_not_evaluable_no_crash(self) -> None:
        engine = MetricsIngestionEngine()
        post = _post(
            "p", T0, views=0, impressions=100, w3=0, w50=0, comp=0,
            likes=0, comments=0, shares=0, saves=0,
        )
        report = engine.calculate_retention_to_engagement(post)
        assert report["evaluable"] is False
        assert report["passes"] is False
        assert report["r_3s"] is None
        assert report["r2e_score"] is None
        assert set(report["checks"].values()) == {False}


class TestConceptValidation:
    """The consecutive rule is chronological over the LAST N uploads (B7)."""

    def test_old_window_passes_but_latest_fails(self) -> None:
        engine = MetricsIngestionEngine()
        posts = [
            _post("p1", T0),
            _post("p2", T0 + timedelta(days=1)),
            _post("p3", T0 + timedelta(days=2)),
            _failing_post("p4", T0 + timedelta(days=3)),
        ]
        # p1..p3 is a passing window, but the LATEST 3 are p2..p4 and p4 fails.
        assert engine.evaluate_concept_validation("c1", posts) is False

    def test_last_three_pass_greenlights_regardless_of_input_order(self) -> None:
        engine = MetricsIngestionEngine()
        posts = [
            _post("p4", T0 + timedelta(days=3)),
            _failing_post("p1", T0),
            _post("p2", T0 + timedelta(days=1)),
            _post("p3", T0 + timedelta(days=2)),
        ]
        # Shuffled caller order: sorting by posted_at must be authoritative.
        assert engine.evaluate_concept_validation("c1", posts) is True

    def test_fewer_posts_than_required_never_greenlights(self) -> None:
        engine = MetricsIngestionEngine()
        posts = [_post("p1", T0), _post("p2", T0 + timedelta(days=1))]
        assert engine.evaluate_concept_validation("c1", posts) is False

    def test_other_concept_posts_are_ignored(self) -> None:
        engine = MetricsIngestionEngine()
        posts = [
            _post("p1", T0),
            _post("p2", T0 + timedelta(days=1)),
            _post("p3", T0 + timedelta(days=2), concept_id="other"),
        ]
        assert engine.evaluate_concept_validation("c1", posts) is False

    def test_zero_view_post_in_window_blocks_without_crash(self) -> None:
        engine = MetricsIngestionEngine()
        posts = [
            _post("p1", T0),
            _post("p2", T0 + timedelta(days=1)),
            _post(
                "p3", T0 + timedelta(days=2), views=0, impressions=100,
                w3=0, w50=0, comp=0, likes=0, comments=0, shares=0, saves=0,
            ),
        ]
        assert engine.evaluate_concept_validation("c1", posts) is False


class TestTheoryMining:
    """Lexical clustering (audit M4) — deterministic, no embedding model."""

    COMMENTS = [
        "the fox spirit is the masked mentor guarding the shrine",
        "calling it: the fox spirit is the masked mentor from the shrine",
        "fox spirit mentor shrine theory is real",
        "animation quality is insane this week",
        "the animation is insane quality wise",
        "wow",
    ]

    def test_repeated_theories_cluster_and_rank_by_volume(self) -> None:
        engine = MetricsIngestionEngine()
        theories = engine.mine_audience_lore_theories(self.COMMENTS, top_n=2)
        assert len(theories) == 2
        assert theories[0] in self.COMMENTS[:3]  # 3-member fox cluster ranks first
        assert theories[1] in self.COMMENTS[3:5]  # 2-member animation cluster second

    def test_mining_is_deterministic(self) -> None:
        engine = MetricsIngestionEngine()
        first = engine.mine_audience_lore_theories(self.COMMENTS, top_n=5)
        second = engine.mine_audience_lore_theories(list(self.COMMENTS), top_n=5)
        assert first == second

    def test_top_n_zero_returns_empty(self) -> None:
        engine = MetricsIngestionEngine()
        assert engine.mine_audience_lore_theories(self.COMMENTS, top_n=0) == []


class TestMockFetcher:
    """The mock generator must be deterministic and invariant-safe."""

    def test_mock_fetch_is_deterministic(self) -> None:
        engine = MetricsIngestionEngine()
        first = engine.fetch_mock_platform_data("post-x", "c1")
        second = engine.fetch_mock_platform_data("post-x", "c1")
        assert first == second  # frozen dataclass equality over every field

    def test_mock_comments_actually_cluster(self) -> None:
        engine = MetricsIngestionEngine()
        pooled = [
            comment
            for index in range(3)
            for comment in engine.fetch_mock_platform_data(f"p{index}", "c1").comments_text_list
        ]
        theories = engine.mine_audience_lore_theories(pooled, top_n=3)
        assert theories  # thematically repeated fan-theory comments must surface


class TestWebhook:
    """Webhook validation paths that need no network."""

    async def test_non_https_url_rejected(self) -> None:
        engine = MetricsIngestionEngine()
        with pytest.raises(WebhookDispatchError):
            await engine.dispatch_greenlight_webhook(
                "c1", {}, "http://discord.com/api/webhooks/x/y"
            )

    def test_payload_shaped_per_host(self) -> None:
        build = MetricsIngestionEngine._build_webhook_payload
        discord = build("c1", {"k": "v"}, "discord.com")
        assert "content" in discord and "embeds" in discord
        slack = build("c1", {"k": "v"}, "hooks.slack.com")
        assert "text" in slack and "content" not in slack
        generic = build("c1", {"k": "v"}, "example.com")
        assert generic == {"concept_id": "c1", "summary": {"k": "v"}}
