"""Contact location, relative sliding, temporal roles and negative evidence."""
from copy import deepcopy
import numpy as np
import pytest
from src.back_contact import directional_back_contacts, BackContactTracker, ROLES
from src.rule_classifier import WashHandRuleClassifier
from test_palm_priority import flat_pair


def hand():
    h = np.zeros((21, 3), dtype=float)
    for base, x in zip((5, 9, 13, 17), (-0.45, -0.15, 0.15, 0.45)):
        for i in range(4):
            h[base+i] = [x, 1+i*0.25, 0]
    return h


def evidence(role=ROLES[0], score_contact=0.95, moving=True, region="central"):
    f = flat_pair()
    f.update(shape_speed=0.5, observed_sides=["left", "right"])
    f['inter_hand']['back_contacts'] = [dict(
        role=role, region=region, contact=score_contact, speed=1.0 if moving else 0,
        alignment=1.0 if moving else 0, asymmetry=0.8, has_previous=True,
    )]
    return f


def test_sliding_speed_is_fps_invariant_and_directional():
    h = hand()
    speeds = []
    for fps in (15, 30, 60):
        c = directional_back_contacts(h+[0, 0.5/fps, 0], h, h, h, 1/fps)[0]
        speeds.append(c['speed'])
        assert c['alignment'] == pytest.approx(1)
    np.testing.assert_allclose(speeds, 0.5)
    transverse = directional_back_contacts(h+[0.01, 0, 0], h, h, h)[0]
    assert transverse['alignment'] == pytest.approx(0)


def test_common_camera_motion_and_independent_depth_origins_do_not_create_sliding():
    left, right = hand()+[0.1, 0.1, 0], hand()
    a = 0.4
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0,0,1]])
    current_left = left @ rot.T * 1.3 + [3, 2, 7]
    current_right = right @ rot.T * 1.3 + [3, 2, -9]
    for c in directional_back_contacts(current_left, current_right, left, right):
        assert c['speed'] < 1e-10


@pytest.mark.parametrize('offset,region', [(-0.6,'wrist'),(0,'central'),(0.8,'fingers')])
def test_contact_regions(offset, region):
    h = hand()
    assert directional_back_contacts(h+[0,offset,0],h)[0]['region'] == region
    assert directional_back_contacts(h+[2,0,0],h)[0]['region'] == 'off_hand'


@pytest.mark.parametrize('fps', [15,30,60])
def test_sustained_role_and_motion_required_for_credit(fps):
    tracker = BackContactTracker()
    for i in range(fps):
        r = tracker.update(evidence(), i/fps, 1/fps, True, True)
        if i/fps < 0.40:
            assert not r['credit_ready']
    assert r['credit_ready']
    changed = tracker.update(evidence(role=ROLES[1]),1,1/fps,True,True)
    assert changed['role'] == ROLES[1]
    assert not changed['credit_ready']


def test_role_near_tie_does_not_flip_each_frame():
    tracker = BackContactTracker()
    tracker.update(evidence(),0,1/30,True,True)
    f = evidence()
    c = deepcopy(f['inter_hand']['back_contacts'][0])
    c.update(role=ROLES[1], contact=1)
    f['inter_hand']['back_contacts'].append(c)
    assert tracker.update(f,1/30,1/30,True,True)['role'] == ROLES[0]


def test_held_or_reacquired_coordinates_cannot_start_contact_evidence():
    for observed, prev_observed in ((False,True),(True,False),(False,False)):
        tracker = BackContactTracker()
        for i in range(50):
            r = tracker.update(evidence(),i/30,1/30,observed,prev_observed)
        assert r['score'] == 0
        assert not r['credit_ready']


def test_single_observation_can_briefly_continue_but_cannot_extend_motion_history():
    tracker = BackContactTracker()
    for i in range(30):
        tracker.update(evidence(),i/30,1/30,True,True)
    f = evidence(); f.update(has_right=False,observed_sides=['left'])
    f['inter_hand']['back_contacts']=[]
    r = tracker.update(f,1,1/30,False,True)
    assert r['carried'] and r['credit_ready']
    r = tracker.update(f,1.2,0.2,False,False)
    assert not r['credit_ready'] and r['score']==0


def test_static_contact_and_visible_contradiction_cannot_receive_credit():
    tracker = BackContactTracker()
    for i in range(30):
        r=tracker.update(evidence(moving=False),i/30,1/30,True,True)
    assert r['score']==0 and not r['credit_ready']
    tracker.update(evidence(),1,1/30,True,True)
    r=tracker.update(evidence(region='wrist'),1.03,0.03,True,True)
    assert r['score']==0 and not r['credit_ready']


def test_occlusion_with_curled_visible_hand_does_not_carry_outside():
    tracker=BackContactTracker()
    tracker.update(evidence(),0,1/30,True,True)
    f=evidence();f.update(left_angles=[80]*5,has_right=False,observed_sides=['left'])
    f['inter_hand']['back_contacts']=[]
    assert tracker.update(f,.03,.03,False,True)['score']==0


def test_long_gap_resets_role_and_motion_history():
    tracker=BackContactTracker()
    for i in range(30):
        tracker.update(evidence(),i/30,1/30,True,True)
    r=tracker.update(evidence(),3,2,True,False)
    assert r['role'] is None and not r['credit_ready']


def test_supported_back_contact_can_outweigh_noisy_opposing_normals():
    f=flat_pair(palm_dot=-.9)
    f['back_evidence']={'score':.7}
    p=WashHandRuleClassifier().predict_probabilities(f)
    assert max(p,key=p.get)=='outside'
    assert p['outside']>p['inside']


def test_single_flat_hand_remains_weak_inside_candidate():
    f={'has_left':True,'active_angles':[160]*5,'active_four_finger_spread':.4}
    p=WashHandRuleClassifier().predict_probabilities(f)
    assert max(p,key=p.get)=='inside'
    assert .30<p['inside']<.35


def test_normals_without_sliding_are_only_weak_outside_candidate():
    p=WashHandRuleClassifier().predict_probabilities(flat_pair(palm_dot=.9))
    assert p['outside']<.5


def test_stale_second_hand_cannot_upgrade_inside_to_strong_contact():
    f=flat_pair();f.update(contains_held=True,both_hands_observed=False)
    p=WashHandRuleClassifier().predict_probabilities(f)
    assert max(p,key=p.get)=='inside'
    assert p['inside']<.45


def test_recent_back_evidence_outweighs_held_inside_candidate():
    f=flat_pair();f.update(contains_held=True,both_hands_observed=False,
                         back_evidence={'score':.7})
    assert WashHandRuleClassifier().predict(f)[0]=='outside'
