"""Palm contact and finger interlacing must supply different evidence."""
import numpy as np
from src.rule_classifier import WashHandRuleClassifier


def flat_pair(palm_dot=-0.8, distance=0.9, angle=45):
    left=np.zeros((21,3));right=left.copy()
    left[9]=[0,1,0]
    right[9]=[np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle)),0]
    return {'has_left':True,'has_right':True,'both_hands_detected':True,
            'left_norm':left,'right_norm':right,'palm_normal_dot':palm_dot,
            'left_angles':[160]*5,'right_angles':[170]*5,
            'inter_hand':{'palm_center_dist':distance,'wrist_dist':distance+0.5,
                          'interlace_depth':0.8,'finger_x_overlap':0.75,
                          'left_four_finger_spread':0.4,'right_four_finger_spread':0.4,
                          'interlace_alternations':1}}


def test_opposing_palms_no_longer_require_narrow_axis_or_wrist_spacing():
    p=WashHandRuleClassifier().predict_probabilities(flat_pair())
    assert max(p,key=p.get)=='inside'
    assert p['inside']>0.7


def test_spread_and_overlap_without_alternation_are_not_interlace():
    assert WashHandRuleClassifier().predict(flat_pair())[0]=='inside'


def test_crossed_fingers_can_outweigh_opposing_palms():
    f=flat_pair();f['inter_hand']['interlace_alternations']=5
    assert WashHandRuleClassifier().predict(f)[0]=='interlace'


def test_same_facing_palms_are_not_automatically_inside():
    assert WashHandRuleClassifier().predict(flat_pair(palm_dot=0.8))[0]=='outside'


def test_far_apart_hands_are_not_palm_contact():
    assert WashHandRuleClassifier().predict(flat_pair(distance=3))[0]=='other'


def test_single_flat_hand_wide_thumb_is_inside_candidate_not_interlace():
    f={'has_left':True,'has_right':False,'both_hands_detected':False,
       'active_angles':[151,170,170,170,170], 'active_spread':0.55,
       'active_four_finger_spread':0.3}
    assert WashHandRuleClassifier().predict(f)[0]=='inside'


def test_no_hands_remains_other():
    assert WashHandRuleClassifier().predict({})[0]=='other'
