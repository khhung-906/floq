from agents.fql import FQLAgent
from agents.fql_iqn import FQLIQNAgent
from agents.floq import FloQAgent
from agents.tql_pac_fql_actor import PACFQLActorAgent


agents = dict(
    fql=FQLAgent,
    fqliqn = FQLIQNAgent,
    floq = FloQAgent,
    pac_fql_actor=PACFQLActorAgent,
)