"""
Make cron_expression non-nullable

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2024-12-15 12:30:00.000000

"""

from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6g7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # First, update any existing NULL values to a default cron expression
    # This handles any legacy data that might have NULL cron_expression
    op.execute("UPDATE tasks SET cron_expression = '0 0 * * *' WHERE cron_expression IS NULL")

    # Then make the column non-nullable
    op.alter_column("tasks", "cron_expression", nullable=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Make the column nullable again
    op.alter_column("tasks", "cron_expression", nullable=True)
    # ### end Alembic commands ###
