"""Re-export inline tests from agency/common_skills/ so that `pytest tests/` picks them up."""
from agency.common_skills.summariser import *       # noqa: F401, F403
from agency.common_skills.writer import *           # noqa: F401, F403
from agency.common_skills.find_papers import *      # noqa: F401, F403
from agency.common_skills.summarise_paper import *  # noqa: F401, F403
from agency.common_skills.compile_report import *   # noqa: F401, F403
