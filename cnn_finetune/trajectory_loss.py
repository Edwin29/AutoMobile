"""Experimental outcome-labelled trajectory preference objective.

This is a continuous-action, mean-log-ratio preference surrogate, NOT a
differentiable vehicle simulator or a guarantee of policy improvement. Labels
come from paired closed-loop rollouts. No teacher output appears in the core
objective. Keep reference policy frozen and re-collect after policy updates.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CostConfig:
    failure: float = 100.
    unrecovered: float = 20.
    outside: float = 10.
    clearance: float = 2.
    progress: float = 1.
    safe_margin_cm: float = 1.
    risk_width_cm: float = 3.
    target_speed_cm_s: float = 4.


def rollout_cost(*, horizon_s, observed_s, outside_s, risk_integral_s,
                 progress_cm, remaining_cm, failed, completed, end_margin_cm,
                 config=CostConfig()):
    """Fixed-horizon cost with absorbing failure; completion needs verified lap.

    A failed run cannot improve its score by ending early. A valid completed
    lap is a zero-running-cost absorbing state. Progress is local signed route
    progress, never Euclidean closeness to a destination. Risk integral uses
    clip((1cm-margin)/3cm,0,1), integrated over physical time.
    """
    values=[horizon_s,observed_s,outside_s,risk_integral_s,progress_cm,remaining_cm,end_margin_cm]
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite rollout metrics')
    if horizon_s<=0 or not 0<=observed_s<=horizon_s+1e-6:raise ValueError('Invalid horizon')
    if not 0<=outside_s<=observed_s+1e-6 or not 0<=risk_integral_s<=observed_s+1e-6:
        raise ValueError('Invalid integrals')
    if remaining_cm<0:raise ValueError('Negative remaining route')
    if failed and completed:raise ValueError('Conflicting terminal outcomes')
    missing=max(0.,horizon_s-observed_s)
    if not (failed or completed) and missing>.04:
        raise ValueError('Censored rollout requires explicit terminal status')
    if failed:
        outside_s+=missing;risk_integral_s+=missing
    target=min(config.target_speed_cm_s*horizon_s,max(remaining_cm,1e-6))
    deficit=0. if completed else 1.-max(0.,min(1.,progress_cm/target))
    components=dict(failure=config.failure*float(failed),
        unrecovered=config.unrecovered*float(not completed and end_margin_cm<0),
        outside=config.outside*outside_s/horizon_s,
        clearance=config.clearance*risk_integral_s/horizon_s,
        progress=config.progress*deficit)
    return dict(total=sum(components.values()),components=components)


def _trajectory_ratio(mean, action, reference, mask, sigma):
    if mean.shape!=action.shape or mean.shape!=reference.shape or mean.ndim!=3 or mean.shape[-1]!=2:
        raise ValueError('Expected matching [batch,time,2] tensors')
    if mask.shape!=mean.shape[:2] or torch.any(mask.sum(-1)<=0):
        raise ValueError('Every trajectory requires a nonempty mask')
    if torch.any(mask<0) or torch.any(mask>1):raise ValueError('Invalid mask')
    for value in (mean,action,reference,mask):
        if not torch.isfinite(value).all():raise ValueError('Nonfinite training input')
    action=action.detach();reference=reference.detach();mask=mask.detach()
    # Untruncated fixed-variance Gaussian is a scoring surrogate in normalized
    # network coordinates. It is NOT the density of clipped/dithered motor PWM.
    ratio=(.5*((reference-action)/sigma).square()-.5*((mean-action)/sigma).square()).sum(-1)
    count=mask.sum(-1)
    score=(ratio*mask).sum(-1)/count
    kl=(.5*((mean-reference)/sigma).square().sum(-1)*mask).sum(-1)/count
    return score,kl


def trajectory_preference_loss(winner_mean, winner_actions, winner_reference, winner_mask,
                               loser_mean, loser_actions, loser_reference, loser_mask,
                               *, beta=1., anchor_weight=.1, sigma=(.15,.20), weights=None):
    """Prefer the lower closed-loop-cost sequence, with a frozen-policy anchor.

    Inputs contain ONLY the equal-duration intervention, not its common frozen
    continuation. Different images on each branch must be kept: replaying the
    winner's actions against the loser's images would be invalid.
    """
    if beta<=0 or anchor_weight<0 or any(v<=0 for v in sigma):raise ValueError('Invalid loss configuration')
    std=winner_mean.new_tensor(sigma)
    w,wkl=_trajectory_ratio(winner_mean,winner_actions,winner_reference,winner_mask,std)
    l,lkl=_trajectory_ratio(loser_mean,loser_actions,loser_reference,loser_mask,std)
    if w.shape!=l.shape:raise ValueError('Unpaired batches')
    per=F.softplus(-beta*(w-l))
    anchor=.5*(wkl+lkl)
    if weights is None:weights=torch.ones_like(per)
    weights=weights.detach()
    if weights.shape!=per.shape or not torch.isfinite(weights).all() or torch.any(weights<0) or weights.sum()<=0:
        raise ValueError('Invalid pair weights')
    loss=((per+anchor_weight*anchor)*weights).sum()/weights.sum()
    return loss,dict(preference=(per*weights).sum()/weights.sum(),anchor=(anchor*weights).sum()/weights.sum(),
                     preference_margin=((w-l)*weights).sum()/weights.sum())
