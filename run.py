from src.run_tracker import run_tracker
from src.particle_tracker import ParticleParams
parameters = ParticleParams()
print(run_tracker("./data", "car", parameters))