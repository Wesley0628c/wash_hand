"""
Wash Hand Finite State Machine (FSM) Module
Manages the lifecycle, state transitions, state priors, and timing requirements
of 7-step hand washing in both Sequential (Guided) and Adaptive (Free) modes.
"""

import time
from typing import List, Dict, Set, Optional, Tuple, Union
import numpy as np

STEPS_ORDER = [
    "inside",
    "outside",
    "interlace",
    "knuckles",
    "thumb",
    "fingertips",
    "wrist",
]

STEPS_ZH = {
    "inside": "內",
    "outside": "外",
    "interlace": "夾",
    "knuckles": "弓",
    "thumb": "大",
    "fingertips": "立",
    "wrist": "腕",
}

STEPS_DESC = {
    "inside": "掌心對掌心搓洗",
    "outside": "掌心搓洗手背",
    "interlace": "十指交錯搓洗指縫",
    "knuckles": "指背搓洗掌心",
    "thumb": "旋轉搓洗大拇指",
    "fingertips": "指尖搓洗掌心",
    "wrist": "旋轉搓洗手腕",
}


class WashHandStateMachine:
    """
    Finite State Machine (FSM) for 7-Step Hand Washing Detection.
    Features:
      - Formal States: IDLE, INSIDE, OUTSIDE, INTERLACE, KNUCKLES, THUMB, FINGERTIPS, WRIST, COMPLETED.
      - Self-Loop Hysteresis: Emits state prior probabilities to prevent transient prediction flickers.
      - Progress Retention (Non-destructive Pause): Freezes timers on occlusion or dropout instead of wiping progress.
      - Dual Transition Modes: Strict Sequential Guidance ('sequence') and Adaptive Smart Flow ('free').
    """

    def __init__(
        self,
        mode: str = "sequence",  # "sequence" (default guided order) or "free" (any order)
        step_duration: float = 1.0,  # required reliable seconds per step
        error_tolerance: float = 1.5,  # grace window before marking paused
    ):
        self.mode = mode
        self.step_duration = step_duration
        self.error_tolerance = error_tolerance

        self.current_step_idx = 0
        self.completed_steps: Set[str] = set()
        self.step_times: Dict[str, float] = {step: 0.0 for step in STEPS_ORDER}
        self.observed_times: Dict[str, float] = {step: 0.0 for step in STEPS_ORDER}
        self.contiguous_times = {step: 0.0 for step in STEPS_ORDER}
        self.evidence_gaps = {step: 0.0 for step in STEPS_ORDER}

        self.active_step: Optional[str] = None
        self.step_timer = 0.0
        self.grace_timer = 0.0
        self.is_paused = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.is_completed = False

    def start(self):
        """Start or reset the state machine for a session."""
        self.reset()

    def reset(self):
        """Reset state machine for a new wash session."""
        self.current_step_idx = 0
        self.completed_steps.clear()
        self.step_times = {step: 0.0 for step in STEPS_ORDER}
        self.observed_times = {step: 0.0 for step in STEPS_ORDER}
        self.contiguous_times = {step: 0.0 for step in STEPS_ORDER}
        self.evidence_gaps = {step: 0.0 for step in STEPS_ORDER}
        self.active_step = None
        self.step_timer = 0.0
        self.grace_timer = 0.0
        self.is_paused = False
        self.start_time = None
        self.end_time = None
        self.is_completed = False

    @property
    def current_state(self) -> str:
        """Return the current formal FSM state string."""
        if self.is_completed:
            return "COMPLETED"
        if self.start_time is None and self.step_timer == 0.0:
            return "IDLE"
        target = self._get_target_step()
        if target:
            return target.upper()
        if self.active_step:
            return self.active_step.upper()
        return "IDLE"

    def get_state_prior(self) -> Dict[str, float]:
        """
        Return neutral priors: completion state is separate from recognition.
        """
        # Guidance and completion must never bias recognition of the same pixels.
        return {s: 1.0 for s in [*STEPS_ORDER, "other"]}

    def update(
        self,
        detected_label: str,
        dt: float,
        is_observed: Union[bool, float] = True,
        observation_weight: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Update state machine with the latest detected action label and delta time.
        Supports binary observation (bool) or tiered observation weight (0.0 to 1.0).
        Returns (is_step_just_completed, completed_step_name).
        """
        if observation_weight is not None:
            weight = float(np.clip(observation_weight, 0.0, 1.0))
        elif isinstance(is_observed, (int, float)) and not isinstance(is_observed, bool):
            weight = float(np.clip(is_observed, 0.0, 1.0))
        else:
            weight = 1.0 if is_observed else 0.0

        dt = max(0.0, float(dt))
        effective_dt = dt * weight
        for step in STEPS_ORDER:
            if detected_label == step and effective_dt > 0:
                self.evidence_gaps[step] = 0.0
                self.contiguous_times[step] += effective_dt
            else:
                self.evidence_gaps[step] += dt
                if self.evidence_gaps[step] > self.error_tolerance:
                    self.contiguous_times[step] = 0.0
        if self.mode == "sequence":
            target = self._get_target_step()
            if target and self.evidence_gaps[target] > self.error_tolerance:
                self.step_timer = 0.0

        if self.start_time is None and detected_label in STEPS_ORDER and weight > 0.0:
            self.start_time = time.time()

        if detected_label in STEPS_ORDER:
            self.step_times[detected_label] += dt
            if effective_dt > 0.0:
                self.observed_times[detected_label] += effective_dt

        just_completed = False
        completed_step_name = None

        # ── 1. Sequence Mode (Formal Guided FSM) ──────────────────────────
        if self.mode == "sequence":
            if self.is_completed:
                return False, None

            target_step = self._get_target_step()
            if target_step is None:
                self._finish_session()
                return False, None

            is_valid_action = (detected_label == target_step)

            if is_valid_action:
                self.active_step = target_step
                self.is_paused = False
                self.grace_timer = 0.0

                if is_valid_action and effective_dt > 0.0:
                    self.step_timer += effective_dt

                # Check if current step has satisfied target duration
                if self.step_timer >= self.step_duration - 1e-5:
                    self.completed_steps.add(target_step)
                    just_completed = True
                    completed_step_name = target_step
                    self.step_timer = 0.0
                    self.active_step = None
                    self.current_step_idx += 1

                    if self.current_step_idx >= len(STEPS_ORDER):
                        self._finish_session()
            else:
                # Off-target action or temporary occlusion:
                # FSM enters PAUSED state. Progress is NON-DESTRUCTIVELY PRESERVED!
                self.grace_timer += dt
                if self.grace_timer > self.error_tolerance:
                    self.is_paused = True
                    self.active_step = None

        # ── 2. Free Mode (Adaptive Smart FSM) ─────────────────────────────
        else:
            if detected_label in STEPS_ORDER:
                self.active_step = detected_label
                self.is_paused = False
                self.grace_timer = 0.0

                if detected_label not in self.completed_steps:
                    self.step_timer = self.contiguous_times[detected_label]
                    if self.step_timer >= self.step_duration - 1e-5:
                        self.completed_steps.add(detected_label)
                        just_completed = True
                        completed_step_name = detected_label
                        self.step_timer = 0.0

                        if len(self.completed_steps) >= len(STEPS_ORDER) and not self.is_completed:
                            self._finish_session()
            else:
                self.grace_timer += dt
                if self.grace_timer > self.error_tolerance:
                    self.is_paused = True
                    self.active_step = None

        return just_completed, completed_step_name

    def _get_target_step(self) -> Optional[str]:
        if self.mode == "sequence":
            if self.current_step_idx < len(STEPS_ORDER):
                return STEPS_ORDER[self.current_step_idx]
            return None
        return None

    def _finish_session(self):
        self.is_completed = True
        self.end_time = time.time()

    def get_progress_summary(self) -> Dict:
        """Return structured summary of current wash session status."""
        total_time = 0.0
        if self.start_time is not None:
            end = self.end_time if self.end_time is not None else time.time()
            total_time = max(0.0, end - self.start_time)

        target = self._get_target_step()
        if self.mode == "sequence":
            current_progress = min(1.0, self.step_timer / max(0.01, self.step_duration)) if target else 1.0
        else:
            current_progress = min(1.0, self.step_timer / max(0.01, self.step_duration)) if self.active_step else 0.0

        step_progresses = {
            step: (1.0 if step in self.completed_steps else min(1.0, self.contiguous_times[step] / max(0.01, self.step_duration)))
            for step in STEPS_ORDER
        }

        return {
            "mode": self.mode,
            "current_state": self.current_state,
            "target_step": target,
            "active_step": self.active_step,
            "is_paused": self.is_paused,
            "current_step_progress": current_progress,
            "step_progresses": step_progresses,
            "step_timer": self.step_timer,
            "step_duration": self.step_duration,
            "completed_steps": list(self.completed_steps),
            "completed_count": len(self.completed_steps),
            "total_steps": len(STEPS_ORDER),
            "is_completed": self.is_completed,
            "total_time": total_time,
            "step_times": self.step_times,
            "observed_times": self.observed_times,
            "contiguous_times": self.contiguous_times,
        }

    def get_fsm_feedback(self, detected_label: str, confidence: Optional[float] = None) -> str:
        """Generate context-aware FSM guidance and feedback message."""
        if self.is_completed:
            return "🎉 恭喜！七步洗手已全部標準完成！"

        conf_str = f"（信心度 {int(confidence * 100)}%）" if (confidence is not None and confidence > 0.0) else ""

        target = self._get_target_step()
        if self.mode == "sequence" and target:
            target_zh = STEPS_ZH.get(target, target)
            target_desc = STEPS_DESC.get(target, "")
            step_num = self.current_step_idx + 1

            if detected_label == target:
                return f"步驟 {step_num}/7 正確：【{target_zh}】{target_desc}{conf_str}"
            elif detected_label in STEPS_ORDER:
                det_zh = STEPS_ZH.get(detected_label, detected_label)
                return f"目前步驟為 {step_num}/7【{target_zh}】，請進行【{target_zh}】搓洗（非【{det_zh}】）"
            else:
                return f"請將雙手就位，進行步驟 {step_num}/7：【{target_zh}】{target_desc}"
        else:
            if detected_label in STEPS_ORDER:
                det_zh = STEPS_ZH.get(detected_label, detected_label)
                det_desc = STEPS_DESC.get(detected_label, "")
                return f"進行中：【{det_zh}】{det_desc}{conf_str}"
            return "請就位雙手，依洗手七字訣（內外夾弓大立腕）進行搓洗"
