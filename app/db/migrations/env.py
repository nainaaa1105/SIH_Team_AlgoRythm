from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.db.models import Base

# Members 2 and 3 declare their tables against this same Base. Importing
# them here registers them on the shared MetaData so
# `alembic revision --autogenerate` sees the whole schema, not just M1's
# share. Each is guarded because M1's containers can run perfectly well
# without the other members' packages installed.
try:
    import classifier.db.models  # noqa: F401  (M2: cluster_features, classifications, training_labels)
except ImportError:
    pass

try:
    import geospatial.db.models  # noqa: F401  (M3: attributions, plumes, corridors, routes)
except ImportError:
    pass

try:
    import imagery.db.models  # noqa: F401  (M4: patches, image predictions, thermal retrievals)
except ImportError:
    pass

try:
    import temporal.db.models  # noqa: F401  (M5: PTSI registry, rhythm fingerprints, Kalman states)
except ImportError:
    pass

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
