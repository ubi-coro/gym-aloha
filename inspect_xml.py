import mujoco
from mujoco import viewer

MODEL = "/home/jzilke/ws/gym-aloha/assets/ur5e_gripper/scene.xml"

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)

with viewer.launch_passive(m, d) as v:
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
