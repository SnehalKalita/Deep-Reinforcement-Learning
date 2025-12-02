
from ReflexCtrInterface import MyoLegReflex
r = MyoLegReflex()
env = r.env   # underlying myosuite env
sim = env.sim

print("Body names:")
print(sim.model.body_names)            # bytes array
print("Geom names:")
print(sim.model.geom_names)
print("Sensor names:")
try:
    print(sim.model.sensor_names)
except Exception:
    # older mujoco versions have different attr names
    import numpy as _np
    s=[]
    for i in range(sim.model.nsensor):
        s.append(sim.model.sensor_id2name(i))
    print(s)
# close env if possible
try: env.close()
except: pass

