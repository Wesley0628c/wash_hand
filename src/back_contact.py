"""Image-plane palm-region contact and temporal dorsum-rubbing evidence.

Landmarks describe a skeleton, not skin surfaces. ``surface`` is an orientation
estimate; independent wrist-relative z coordinates cannot establish contact.
"""
from collections import deque
import numpy as np

MCPS = [5, 9, 13, 17]
ROLES = ("left_palm_right_back", "right_palm_left_back")


def _local_contact(active, target):
    mcp = target[MCPS, :2].mean(axis=0)
    axis = mcp - target[0, :2]
    length = float(np.linalg.norm(axis))
    width = float(np.linalg.norm(target[5, :2] - target[17, :2]))
    if min(length, width) < 1e-5:
        return None
    axis /= length
    lateral = np.array([-axis[1], axis[0]])
    center = (active[0, :2] + active[MCPS, :2].mean(axis=0)) / 2
    delta = center - target[0, :2]
    return np.array([delta @ lateral / length, delta @ axis / length]), width / length


def directional_back_contacts(left, right, prev_left=None, prev_right=None, dt=1/30):
    """Measure each role in the target's rotating/scaling coordinate frame.

    Common translation, rotation and zoom produce zero sliding. Motion is
    measured in projected palm lengths/second and is independent of frame rate.
    """
    if left is None or right is None:
        return []
    contacts = []
    for role, active, target, prev_active, prev_target in (
        (ROLES[0], left, right, prev_left, prev_right),
        (ROLES[1], right, left, prev_right, prev_left),
    ):
        local = _local_contact(active, target)
        if local is None:
            continue
        position, width = local
        x, y = map(float, position)
        region = ("wrist" if y < 0.15 else "fingers" if y > 1.15
                  else "off_hand" if abs(x) > width * 0.75 else "central")
        contact = float(np.clip(1 - abs(x) / max(width, 0.1), 0, 1))
        contact *= float(np.clip(1 - abs(y - 0.65) / 0.85, 0, 1))
        previous = (_local_contact(prev_active, prev_target)
                    if prev_active is not None and prev_target is not None else None)
        speed = alignment = asymmetry = 0.0
        if previous is not None and 0 < dt <= 0.20:
            velocity = (position - previous[0]) / dt
            speed = float(np.linalg.norm(velocity))
            alignment = float(abs(velocity[1]) / speed) if speed > 1e-5 else 0.0
            # A supportive clue only: camera movement affects absolute speeds.
            a_speed = np.linalg.norm(active[MCPS, :2].mean(0) - prev_active[MCPS, :2].mean(0))
            t_speed = np.linalg.norm(target[MCPS, :2].mean(0) - prev_target[MCPS, :2].mean(0))
            asymmetry = float(np.clip((a_speed - t_speed) / (a_speed + t_speed + 1e-6), 0, 1))
        contacts.append(dict(role=role, region=region, lateral=x, longitudinal=y,
                             contact=contact, speed=speed, alignment=alignment,
                             asymmetry=asymmetry, has_previous=previous is not None))
    return contacts


class BackContactTracker:
    """Keep role evidence separate from classification/completion history.

    Fresh contact needs observed hands. Brief one-hand occlusion can continue
    established evidence, but held coordinates never create new sliding evidence.
    """
    def __init__(self, min_score=0.60, role_seconds=0.40, entry_seconds=0.50,
                 occlusion_seconds=0.15):
        self.min_score = min_score
        self.role_seconds = role_seconds
        self.entry_seconds = entry_seconds
        self.occlusion_seconds = occlusion_seconds
        self.reset()

    def reset(self):
        self.role = None
        self.role_start = None
        self.last_contact = None
        self.last_score = 0.0
        self.history = deque()

    def update(self, features, timestamp, dt, both_observed, previous_both_observed):
        if dt > 0.20:
            self.reset()
        inter = features.get("inter_hand", {})
        contacts = inter.get("back_contacts", [])
        dot = float(features.get("palm_normal_dot", 0.0))
        angles = [features.get(f"{side}_angles", [0]*5) for side in ("left", "right")]
        curls = [float(np.mean(a[1:])) for a in angles]
        flat = min(curls) > 120 and max(curls) > 150
        # Inside and outside can both interlace. Penalize only opposing palms
        # with substantial alternating-finger overlap, not spread alone.
        interlaced = (inter.get("interlace_alternations", 0) >= 4
                      and inter.get("finger_x_overlap", 0) >= 0.25
                      and inter.get("interlace_depth", 99) < 1.15)
        thumb = (min(curls) < 135 and inter.get("min_palm_to_thumb", 99) < 0.65
                 and inter.get("min_web_to_thumb", 99) < 0.70)
        tips = inter.get("min_fingertip_spread", 99) < 0.30 and inter.get("min_tips_to_palm", 99) < 0.90
        scored = []
        for c in contacts:
            moving = (both_observed and previous_both_observed and c["has_previous"]
                      and 0.04 <= c["speed"] <= 12.0 and c["alignment"] > 0.5)
            surface = float(np.clip((dot + 1) / 2, 0, 1))
            # Symmetric motion of opposing palms weakens dorsum evidence.
            symmetry = max(0.0, -dot) * (1 - c["asymmetry"])
            score = (0.40*c["contact"] + 0.25*float(moving)
                     + 0.20*c["alignment"]*float(moving) + 0.15*c["asymmetry"]
                     + 0.10*surface - 0.12*symmetry)
            score -= 0.25*float(interlaced and dot < -0.3)
            score -= 0.35*float(thumb) + 0.25*float(tips)
            valid = flat and c["region"] == "central" and moving and not thumb
            c.update(score=float(np.clip(score, 0, 1)), moving=bool(moving),
                     surface_estimate=surface, symmetry=symmetry, valid=bool(valid))
            if valid and c["score"] >= self.min_score:
                scored.append(c)
        best = max(scored, key=lambda c: c["score"], default=None)
        # Retain the established role for near-ties, but allow a genuine switch.
        same = next((c for c in scored if c["role"] == self.role), None)
        if same is not None and best["score"] - same["score"] < 0.10:
            best = same
        fresh = best is not None
        if fresh:
            if self.last_contact is None or timestamp-self.last_contact > self.occlusion_seconds or best["role"] != self.role:
                self.history.clear()
                self.role = best["role"]
                self.role_start = timestamp
            self.last_contact = timestamp
            self.last_score = best["score"]
            self.history.append((timestamp, min(dt, 0.20)))
        while self.history and self.history[0][0] < timestamp-0.8:
            self.history.popleft()
        age = timestamp-self.last_contact if self.last_contact is not None else float("inf")
        visible = features.get("has_left", False) or features.get("has_right", False)
        # Briefly retain the display at stroke reversals or occlusion. Neither
        # extends motion history; visible contrary geometry cancels the display.
        same_contact = next((c for c in contacts if c["role"] == self.role
                             and c["region"] == "central"), None)
        observed_sides = features.get("observed_sides", ("left", "right"))
        visible_flat = any(float(np.mean(features.get(f"{side}_angles", [0]*5)[1:])) > 140
                           for side in observed_sides)
        carry = (visible and age <= self.occlusion_seconds
                 and ((not both_observed and visible_flat)
                      or (flat and not thumb and same_contact is not None)))
        if not fresh and not carry:
            score = 0.0
        else:
            score = self.last_score
        if age > self.occlusion_seconds or not visible:
            self.role = None
            self.role_start = None
            self.history.clear()
        seconds = sum(duration for _, duration in self.history)
        role_age = timestamp-self.role_start if self.role_start is not None else 0.0
        current_motion = fresh or (carry and not both_observed
                                   and 0.02 <= features.get("shape_speed", 0.0) <= 12.0)
        credit_ready = bool(score >= self.min_score and role_age >= self.role_seconds
                            and seconds + 1e-9 >= self.entry_seconds and current_motion)
        evidence = dict(score=float(score), role=self.role, fresh=fresh,
                        carried=bool(carry and not fresh), role_seconds=role_age,
                        motion_seconds=seconds, credit_ready=credit_ready)
        features["back_evidence"] = evidence
        return evidence
