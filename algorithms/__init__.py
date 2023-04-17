from .cpo import CPO
from .focops import FOCOPS
from .p3o import P3O
from .pcpo import PCPO
from .ppol import PPOL
from .trpol import TRPOL
from .cppo_pid import CPPO_PID
# from .ppo import PPO
REGISTRY = {
    'cpo': CPO,
    'ppol': PPOL,
    'trpol': TRPOL,
    'focops': FOCOPS,
    'pcpo': PCPO,
    'p3o': P3O,
    'cppo_pid': CPPO_PID,
    # 'ppo': PPO
}
