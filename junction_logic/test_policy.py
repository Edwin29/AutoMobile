import unittest
from policy_reference import Entry, Sign, Choice, Commitment, exits, select


class PolicyTests(unittest.TestCase):
    def test_all_four_central_arrivals_force_right(self):
        for prev in ('01','10','12','21'):
            e=Entry('visit',prev,'11')
            c=select(e,10,Sign('visit','left',10,.99,True),'straight')
            self.assertEqual(c.action,'right')
            self.assertEqual(c.destination,exits(e)['right'])

    def test_t_side_arrival_has_no_right(self):
        e=Entry('visit','00','10')
        self.assertEqual(exits(e),{'left':'11','straight':'20'})
        self.assertEqual(select(e,10,route_action='right').reason,'exit_absent')

    def test_t_stem_has_no_straight(self):
        e=Entry('visit','11','10')
        self.assertEqual(set(exits(e)),{'left','right'})
        self.assertIsNone(select(e,10,route_action='straight').action)

    def test_missing_command_does_not_default_straight(self):
        self.assertIsNone(select(Entry('visit','00','10'),10).action)

    def test_invalid_localization(self):
        self.assertIsNone(select(Entry('visit','00','11'),10).action)

    def test_stale_future_unconfirmed_wrong_event_signs(self):
        e=Entry('visit','00','10')
        for sign in (Sign('visit','left',8,.99,True),Sign('visit','left',11,.99,True),
                     Sign('visit','left',10,.5,True),Sign('old','left',10,.99,True),
                     Sign('visit','left',10,.99,False),Sign('visit','left',10,float('nan'),True)):
            self.assertIsNone(select(e,10,sign,'straight').action)

    def test_valid_sign_overrides_test_route(self):
        e=Entry('visit','00','10')
        self.assertEqual(select(e,10,Sign('visit','left',9.5,.99,True),'straight').action,'left')

    def test_lock_and_verified_exit_and_repeated_event(self):
        p=Commitment();e=Entry('visit','00','10')
        c=select(e,10,route_action='left')
        self.assertFalse(p.commit(e,c,ready=False))
        self.assertTrue(p.commit(e,c,ready=True))
        self.assertFalse(p.commit(e,select(e,10,route_action='straight'),ready=True))
        self.assertIsNone(p.finish('other',verified_exit=True))
        self.assertIsNone(p.finish('visit',verified_exit=False))
        self.assertEqual(p.finish('visit',verified_exit=True),('10','11'))
        self.assertFalse(p.commit(e,c,ready=True))

    def test_central_lock_rejects_forged_straight(self):
        self.assertFalse(Commitment().commit(Entry('v','10','11'),Choice('straight','12','bad'),ready=True))

    def test_every_directed_edge_has_unique_legal_outputs(self):
        from policy_reference import NODES
        count=0
        for a in NODES:
            for b in NODES:
                if sum(abs(x-y) for x,y in zip(NODES[a],NODES[b]))!=1:continue
                out=exits(Entry('v',a,b));count+=1
                self.assertNotIn(a,out.values())
                self.assertEqual(len(out),len(set(out.values())))
        self.assertEqual(count,24)


if __name__=='__main__':unittest.main()
