# shadow hand pick bottle:
1. rigid body: 73
    26*2: shadow hand * 2
    4*5: bottle * 5
    1*1: table * 1
2. dof:
    shadow hand: 24
3. observation:
    0:24: shadow_hand_dof_pos
    24:48: shadow_hand_dof_vel
    48:72: shadow_hand_dof_force
    72:137: finger_state
    137:167: finger_torque
    167:170: hand_pos
    170:173: hand_ori(euler)
    173:199: action
    199:223: another_hand_dof_pos
    223:247: another_hand_dof_vel
    247:271: another_hand_dof_force
    271:336: another_hand_finger_state
    336:366: another_hand_finger_torque
    336:369: another_hand_pos
    369:372: another_hand_ori(euler)
    372:398: another_action

    398:405: bottle2_pose
    405:408: bottle2_linvel
    408:411: bottle2_angvel
    411:414: bottle2_cab
    414:421: bottle4_pose
    421:424: bottle4_linvel
    424:427: bottle4_angvel
    427:430: bottle4_cab

# shadow hand draw blocks:
1. rigid body: 70
    26*2: shadow hand * 2
    17: blocks * 17
    1*1: table * 1
2. dof:
    shadow hand: 24 * 2
3. observation:
    0:24: shadow_hand_dof_pos
    24:48: shadow_hand_dof_vel
    48:72: shadow_hand_dof_force
    72:137: finger_state
    137:167: finger_torque
    167:170: hand_pos
    170:173: hand_ori(euler)
    173:199: action
    199:223: another_hand_dof_pos
    223:247: another_hand_dof_vel
    247:271: another_hand_dof_force
    271:336: another_hand_finger_state
    336:366: another_hand_finger_torque
    336:369: another_hand_pos
    369:372: another_hand_ori(euler)
    372:398: another_action

    398:405: bottle2_pose
    405:408: bottle2_linvel
    408:411: bottle2_angvel
    411:414: bottle2_cab







