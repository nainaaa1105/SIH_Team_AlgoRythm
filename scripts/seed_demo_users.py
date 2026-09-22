"""Seed the three demo accounts the login screen's quick-sign-in buttons
use, so an examiner/reviewer can get into the dashboard in one click
instead of typing credentials: one administrator and two state accounts
(Haryana, Odisha) — the same two states the login screen's demo
buttons are labelled for.

Idempotent: safe to re-run. An existing demo account's password is
reset to the known demo password rather than skipped, so the buttons
keep working even if a demo account was created earlier with a
different password (e.g. by a previous version of this script) or
someone typo'd it into existence with a broken password by hand.

Usage:
    python -m scripts.seed_demo_users
"""
import logging

from app.auth.security import hash_password
from app.db.models import User
from app.db.session import session_scope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Not a secret in the security sense — these are deliberately public,
# printed demo credentials for a reviewer to click through the app.
# Never reuse this pattern for a real account.
DEMO_USERS = [
    {
        "username": "admin_demo",
        "password": "AdminDemo@123",
        "full_name": "Administrator",
        "government_id": "ADMIN-DEMO-0001",
        "role": "admin",
        "state": None,
    },
    {
        "username": "haryana_demo",
        "password": "HaryanaDemo@123",
        "full_name": "Haryana State Officer",
        "government_id": "HR-STATE-DEMO-0001",
        "role": "state",
        "state": "Haryana",
    },
    {
        "username": "odisha_demo",
        "password": "OdishaDemo@123",
        "full_name": "Odisha State Officer",
        "government_id": "OD-STATE-DEMO-0001",
        "role": "state",
        "state": "Odisha",
    },
]

# Demo accounts for states the login screen's buttons no longer point
# at — a stale row here would just be dead demo data left over from a
# previous swap, so each one is removed the next time this script runs
# rather than accumulating. odisha_demo itself came back into DEMO_USERS
# above (Rajasthan -> Odisha), so it must not be listed here even though
# it was retired in an earlier version of this script.
_RETIRED_USERNAMES = ["westbengal_demo", "rajasthan_demo"]


def seed() -> None:
    with session_scope() as session:
        for username in _RETIRED_USERNAMES:
            retired = session.query(User).filter(User.username == username).first()
            if retired is not None:
                session.delete(retired)
                logger.info("Removed retired demo account %s", username)

        for spec in DEMO_USERS:
            user = session.query(User).filter(User.username == spec["username"]).first()
            if user is None:
                user = User(username=spec["username"])
                session.add(user)
                logger.info("Creating demo account %s", spec["username"])
            else:
                logger.info("Refreshing existing demo account %s", spec["username"])

            user.password_hash = hash_password(spec["password"])
            user.full_name = spec["full_name"]
            user.government_id = spec["government_id"]
            user.role = spec["role"]
            user.state = spec["state"]

    logger.info("Demo accounts ready: %s", ", ".join(u["username"] for u in DEMO_USERS))


if __name__ == "__main__":
    seed()
