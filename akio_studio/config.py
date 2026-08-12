"""Central configuration for Akio Studio.

Every tunable threshold, model identifier, and resource budget lives here so
that the six modules never hard-code their own copies. All memory figures are
estimates for an Apple Silicon M4 Pro with a 24 GB unified memory pool and are
used by the pool coordinator's ledger to *gate* model loads, not to enforce OS
limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

#: Master timeline rate. 23.976 fps is exactly 24000/1001 — using the rounded
#: float drifts by one frame roughly every 1000 seconds, which is enough to
#: mis-attribute a retention drop-off to a neighbouring frame in DPO logging.
MASTER_FPS: Fraction = Fraction(24000, 1001)

#: Total unified memory on the target machine, in GiB.
UNIFIED_MEMORY_GIB: float = 24.0

#: Memory the OS + ComfyUI host process + safety margin are assumed to hold.
SYSTEM_RESERVED_GIB: float = 6.0

#: Estimated resident footprints of each loadable model stage, in GiB.
#: qwen2.5-coder:14b at Q4_K_M weighs ~9 GiB plus KV cache; an SDXL-class
#: anime checkpoint with IP-Adapter/ControlNet stack ~10 GiB; a WAN-class
#: video model ~14 GiB. LLM + either diffusion stage cannot co-reside within
#: the ~18 GiB usable pool, which is why stages are gated sequentially.
MODEL_FOOTPRINTS_GIB: dict[str, float] = {
    "llm": 10.0,
    "image_diffusion": 10.0,
    "video_diffusion": 14.0,
    # After Effects / aerender working set for the re-frame + render stage —
    # no model weights, but 16-bpc frame caches are far from free.
    "edit": 8.0,
}


@dataclass(frozen=True)
class GreenlightThresholds:
    """Metric gates a concept must clear to earn a 16:9 episode greenlight.

    Retention ratios are expressed as fractions of *views* (a view is a
    playback start); engagement velocity is per-*impression* and expressed in
    percent, matching how platform analytics report the two families.
    """

    hook_retention_3s: float = 0.70
    midpoint_retention: float = 0.45
    completion_rate: float = 0.35
    engagement_velocity_pct: float = 8.5
    #: The rule is chronological: the most recent N uploads must *all* pass.
    consecutive_uploads_required: int = 3


@dataclass(frozen=True)
class OllamaConfig:
    """How the pool coordinator reaches the local Ollama server.

    The HTTP API is preferred over spawning ``ollama run`` per call: the CLI
    round-trips through the same server anyway, cannot set a system prompt,
    and gives no control over ``keep_alive`` (model residency).
    """

    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5-coder:14b"
    #: Seconds the model stays resident after a call while a stage is active.
    keep_alive_active: str = "10m"
    #: Explicit context window. Ollama's 4k default silently truncates long
    #: lore contexts; 8k costs ~1.5 GiB of KV cache on a 14B GQA model, which
    #: the "llm" stage footprint already accounts for.
    num_ctx: int = 8192
    request_timeout_s: float = 300.0
    #: Absolute CLI path fallback for Dock-launched contexts where PATH is
    #: minimal (Finder does not inherit the user's shell PATH).
    cli_candidates: tuple[str, ...] = (
        "/opt/homebrew/bin/ollama",
        "/usr/local/bin/ollama",
        "ollama",
    )


@dataclass(frozen=True)
class ComfyUIConfig:
    """How the pool coordinator reaches the local ComfyUI server.

    ComfyUI holds the second-largest memory footprint on the machine; its
    ``POST /free`` endpoint is the only way to release it from outside the
    process, so the purge path depends on this address.
    """

    host: str = "http://127.0.0.1:8188"
    request_timeout_s: float = 30.0


@dataclass(frozen=True)
class ComfyQualityGate:
    """Character-consistency parameters for the ComfyUI quality gate.

    ``motion_bucket_*`` is Stable Video Diffusion conditioning and does not
    exist on WAN nodes — it is validated only when an SVD node is actually in
    the graph. WAN motion is governed by denoise, flow shift and frame count.
    """

    ip_adapter_faceid_weight: float = 0.70  # spec range 0.65–0.75
    pulid_weight: float = 0.60
    controlnet_pose_strength: float = 0.75
    controlnet_pose_end_step: float = 0.80
    character_lora_weight: float = 0.75  # spec range 0.70–0.80
    style_lora_weight: float = 0.35  # spec range 0.30–0.40
    max_combined_lora_weight: float = 1.15
    wan_denoise_min: float = 0.25
    wan_denoise_max: float = 0.35
    #: SVD-only conditioning (see class docstring).
    motion_bucket_min: int = 127
    motion_bucket_max: int = 150

    def __post_init__(self) -> None:
        """Validate the gate so out-of-range params raise instead of drifting.

        Spec ranges are enforced at construction (audit minor list / A5): a
        gate object that exists is a gate object that is in range.
        """
        ranges: tuple[tuple[str, float, float, float], ...] = (
            ("ip_adapter_faceid_weight", self.ip_adapter_faceid_weight, 0.65, 0.75),
            ("pulid_weight", self.pulid_weight, 0.0, 1.0),
            ("controlnet_pose_strength", self.controlnet_pose_strength, 0.0, 1.0),
            ("controlnet_pose_end_step", self.controlnet_pose_end_step, 0.0, 1.0),
            ("character_lora_weight", self.character_lora_weight, 0.70, 0.80),
            ("style_lora_weight", self.style_lora_weight, 0.30, 0.40),
        )
        for name, value, lo, hi in ranges:
            if not (lo <= value <= hi):
                raise ValueError(f"{name}={value} outside spec range [{lo}, {hi}]")
        if self.character_lora_weight > self.max_combined_lora_weight:
            raise ValueError(
                "character_lora_weight alone exceeds max_combined_lora_weight"
            )
        if not (0.0 <= self.wan_denoise_min <= self.wan_denoise_max <= 1.0):
            raise ValueError("WAN denoise bounds must satisfy 0 <= min <= max <= 1")
        if self.motion_bucket_min > self.motion_bucket_max:
            raise ValueError("motion_bucket_min must not exceed motion_bucket_max")

    def effective_lora_weights(self) -> tuple[float, float]:
        """Return ``(character, style)`` LoRA weights honoring the 1.15 cap.

        The spec's per-slot ranges allow 0.80 + 0.40 = 1.20, contradicting its
        own cap. The cap is authoritative: the style weight is scaled down.
        """
        char_w = self.character_lora_weight
        style_w = self.style_lora_weight
        if char_w + style_w > self.max_combined_lora_weight:
            style_w = max(0.0, self.max_combined_lora_weight - char_w)
        return char_w, style_w

    def validate_wan_params(self, denoise: float) -> None:
        """Raise ``ValueError`` when a WAN denoise value leaves the gate."""
        if not (self.wan_denoise_min <= denoise <= self.wan_denoise_max):
            raise ValueError(
                f"WAN denoise {denoise} outside quality gate "
                f"[{self.wan_denoise_min}, {self.wan_denoise_max}]"
            )


@dataclass(frozen=True)
class StudioConfig:
    """Top-level configuration object shared by every module."""

    base_dir: Path = field(default_factory=lambda: Path.home() / "AkioStudio" / "productions")
    thresholds: GreenlightThresholds = field(default_factory=GreenlightThresholds)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    comfyui: ComfyUIConfig = field(default_factory=ComfyUIConfig)
    quality_gate: ComfyQualityGate = field(default_factory=ComfyQualityGate)
    short_form_duration_s: float = 15.0
    #: Real, budgetable checkpoints — the spec's "WAN 2.7"/"Anima-Aesthetic"
    #: pins are unverifiable, so model identity is configuration, not code.
    image_model_id: str = "illustrious-xl-anime"
    video_model_id: str = "wan2.1-t2v-1.3b"

    @property
    def usable_memory_gib(self) -> float:
        """Unified memory left for models after the system reservation."""
        return UNIFIED_MEMORY_GIB - SYSTEM_RESERVED_GIB
