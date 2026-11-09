"""Proposal reference: route policy only, no perception/localization/motor control.

Grid x points east, y north. Node '11' is the central four-way junction.
Entry.previous and Entry.node must come from a localized route, not a line gap.
"""
from dataclasses import dataclass
import math

TURNS = frozenset(('straight', 'left', 'right'))
NODES = {f'{x}{y}': (x, y) for x in range(3) for y in range(3)}


@dataclass(frozen=True)
class Entry:
    event_id: str
    previous: str
    node: str


@dataclass(frozen=True)
class Sign:
    event_id: str
    action: str
    captured_at: float
    confidence: float
    temporally_confirmed: bool


@dataclass(frozen=True)
class Choice:
    action: str | None
    destination: str | None
    reason: str


def exits(entry):
    if not entry.event_id or entry.node not in NODES or entry.previous not in NODES:
        raise ValueError('Unlocalized entry')
    x,y = NODES[entry.node]
    px,py = NODES[entry.previous]
    if abs(x-px)+abs(y-py) != 1:
        raise ValueError('Incoming edge absent')
    incoming = (x-px,y-py)
    out = {}
    for node,(xx,yy) in NODES.items():
        if node == entry.previous or abs(xx-x)+abs(yy-y) != 1:
            continue
        cross = incoming[0]*(yy-y)-incoming[1]*(xx-x)
        out['left' if cross>0 else 'right' if cross<0 else 'straight'] = node
    return out


def select(entry, now, sign=None, route_action=None):
    """TTL 1s and confidence .9 are proposal defaults, not calibrated model values.

    A confirmed central node always overrides conflicting sign/route instructions.
    Unknown/invalid information results in no maneuver. No silent straight fallback.
    """
    try:
        allowed = exits(entry)
    except ValueError:
        return Choice(None,None,'localization_required')
    if not math.isfinite(now):
        return Choice(None,None,'invalid_time')
    if entry.node == '11':
        action,reason = 'right','central_rule'
    elif sign is not None:
        if (sign.event_id != entry.event_id or sign.action not in TURNS or
            not math.isfinite(sign.captured_at) or not 0 <= now-sign.captured_at <= 1 or
            not math.isfinite(sign.confidence) or not .9 <= sign.confidence <= 1 or
            not sign.temporally_confirmed):
            return Choice(None,None,'sign_unusable')
        action,reason = sign.action,'sign'
    elif route_action in TURNS:
        action,reason = route_action,'explicit_test_route'
    else:
        return Choice(None,None,'instruction_required')
    if action not in allowed:
        return Choice(None,None,'exit_absent')
    return Choice(action,allowed[action],reason)


class Commitment:
    """Once committed, ignore subsequent sign changes until verified completion.

    ready/verified_exit are integration contracts: this class does NOT verify paths,
    pose, footprint clearance or exit lanes. The proposal specifies those checks.
    """
    def __init__(self):
        self.active = None
        self.completed = set()

    def commit(self, entry, choice, *, ready):
        if self.active is not None or entry.event_id in self.completed or not ready:
            return False
        allowed = exits(entry)
        if choice.action not in allowed or allowed[choice.action] != choice.destination:
            return False
        if entry.node == '11' and choice.action != 'right':
            return False
        self.active = (entry,choice)
        return True

    def finish(self, event_id, *, verified_exit):
        if not self.active or self.active[0].event_id != event_id or not verified_exit:
            return None
        entry,choice = self.active
        self.completed.add(entry.event_id)
        self.active = None
        return entry.node,choice.destination
