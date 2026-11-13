"""Objective semantics and differentiability checks; no vehicle/hardware imports."""
import unittest
import torch
from .trajectory_loss import rollout_cost, trajectory_preference_loss


class TrajectoryLossTests(unittest.TestCase):
    def test_failure_cannot_hide_by_ending_early(self):
        common=dict(horizon_s=15.,progress_cm=0.,remaining_cm=80.,completed=False,end_margin_cm=-12.)
        early=rollout_cost(**common,observed_s=1.,outside_s=1.,risk_integral_s=1.,failed=True)
        late=rollout_cost(**common,observed_s=15.,outside_s=15.,risk_integral_s=15.,failed=True)
        self.assertAlmostEqual(early['total'],late['total'])

    def test_safe_progress_beats_stalling_but_failure_is_worse(self):
        common=dict(horizon_s=15.,observed_s=15.,outside_s=0.,risk_integral_s=0.,remaining_cm=80.,
                    completed=False,end_margin_cm=2.)
        moving=rollout_cost(**common,progress_cm=60.,failed=False)['total']
        stalled=rollout_cost(**common,progress_cm=0.,failed=False)['total']
        failure=rollout_cost(**common,progress_cm=60.,failed=True)['total']
        self.assertLess(moving,stalled);self.assertLess(stalled,failure)

    def test_verified_completion_absorbs_remaining_horizon(self):
        cost=rollout_cost(horizon_s=15.,observed_s=4.,outside_s=0.,risk_integral_s=0.,progress_cm=20.,
            remaining_cm=20.,failed=False,completed=True,end_margin_cm=2.)
        self.assertEqual(cost['total'],0.)

    def test_unmarked_censoring_rejected(self):
        with self.assertRaises(ValueError):
            rollout_cost(horizon_s=15.,observed_s=4.,outside_s=0.,risk_integral_s=0.,progress_cm=20.,
                remaining_cm=80.,failed=False,completed=False,end_margin_cm=2.)

    def make_pair(self,dtype=torch.double):
        ref=torch.tensor([[[.6,.0],[.6,.0]]],dtype=dtype)
        winner=ref+torch.tensor([0.,.15],dtype=dtype)
        loser=ref-torch.tensor([0.,.15],dtype=dtype)
        return ref,winner,loser,torch.ones(1,2,dtype=dtype)

    def test_finite_difference_and_reference_detach(self):
        ref,wa,la,mask=self.make_pair();ref.requires_grad_();wa.requires_grad_();la.requires_grad_()
        w=ref.detach().clone().requires_grad_();l=w.detach().clone().requires_grad_()
        def f(w,l):return trajectory_preference_loss(w,wa,ref,mask,l,la,ref,mask)[0]
        self.assertTrue(torch.autograd.gradcheck(f,(w,l)))
        f(w,l).backward()
        self.assertIsNone(ref.grad);self.assertIsNone(wa.grad);self.assertIsNone(la.grad)
        self.assertLess(w.grad[...,1].mean().item(),0.)

    def test_masked_padding_has_no_gradient_or_value_effect(self):
        ref,wa,la,mask=self.make_pair()
        w=ref.clone().requires_grad_();l=ref.clone().requires_grad_()
        base=trajectory_preference_loss(w,wa,ref,mask,l,la,ref,mask)[0]
        def pad(t,value):return torch.cat([t,torch.full((1,1,2),value,dtype=t.dtype)],dim=1)
        wp=pad(w,99).detach().requires_grad_();lp=pad(l,-99).detach().requires_grad_()
        mp=torch.cat([mask,torch.zeros(1,1,dtype=mask.dtype)],dim=1)
        padded=trajectory_preference_loss(wp,pad(wa,-7),pad(ref,5),mp,lp,pad(la,3),pad(ref,5),mp)[0]
        self.assertAlmostEqual(base.item(),padded.item())
        padded.backward();self.assertEqual(wp.grad[:,2:].abs().sum().item(),0.)

    def test_gradient_step_improves_preference(self):
        ref,wa,la,mask=self.make_pair()
        mean=ref.clone().requires_grad_()
        def f():return trajectory_preference_loss(mean,wa,ref,mask,mean,la,ref,mask)[0]
        before=f();before.backward()
        with torch.no_grad():mean-=.001*mean.grad
        self.assertLess(f().item(),before.item())


if __name__=='__main__':unittest.main()
