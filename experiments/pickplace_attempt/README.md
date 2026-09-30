# Pick-and-Place Attempt (Franka Panda) — Stretch Goal

Attempted a full pick-and-place task (reach cube -> grasp -> lift -> carry -> place)
using a Franka Panda arm with a 2-finger gripper, extending the KUKA reach/obstacle
work in the main project.

## What worked
- Manual IK-scripted pick-and-place (test_manual_pickplace.py) succeeded reliably:
  gripper made real contact, lifted the cube, carried it, and placed it within 2cm
  of the goal. Confirms the physics, contact detection, and environment logic are sound.

## What didn't converge
- RL training (PPO) attempted twice with different reward shaping:
  - Attempt 1 (pickplace_s0): policy learned to grasp reliably (68% of episodes)
    but always dropped the cube before placing -- likely due to a reward-shaping
    flaw (dropping early was mathematically cheaper than carrying).
  - Attempt 2 (pickplace_s1): after fixing the reward, the policy stopped grasping
    almost entirely (98% never grasped) -- a different local optimum, likely
    still under-exploring the full task chain.

## Conclusion
Sparse, multi-phase manipulation tasks like pick-and-place are known to be
significantly harder to train via RL than single-phase reaching, often requiring
larger compute budgets, curriculum learning, or techniques like hindsight
experience replay. This is documented here as an honest limitation. The KUKA
reach and obstacle-avoidance results (see main README) represent the project's
validated core contributions.
