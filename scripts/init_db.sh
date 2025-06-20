#!/bin/bash
# Database initialization script for SmartScrape

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables if .env exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment variables from .env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "Warning: DATABASE_URL is not set. Using default SQLite database."
    export DATABASE_URL="sqlite:///$PROJECT_ROOT/smartscrape.db"
fi

# Create migrations directory if it doesn't exist
if [ ! -d "$PROJECT_ROOT/migrations/versions" ]; then
    echo "Creating migrations directory structure"
    mkdir -p "$PROJECT_ROOT/migrations/versions"
fi

cd "$PROJECT_ROOT"

# Check if Alembic is installed
if ! command -v alembic &> /dev/null; then
    echo "Error: Alembic is not installed. Please install it using: pip install alembic"
    exit 1
fi

# Create initial migration
echo "Creating initial database migration"
alembic revision --autogenerate -m "Initial database schema"

# Run the migration
echo "Running database migration"
alembic upgrade head

# Create admin user if needed
echo "Creating admin user if it doesn't exist"
python -c "
import os
import sys
sys.path.append('$PROJECT_ROOT')
from models.base import get_db_session
from models.user import User
from passlib.hash import bcrypt
import uuid

admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
admin_password = os.environ.get('ADMIN_PASSWORD', 'admin')
admin_email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')

with get_db_session() as session:
    # Check if admin user exists
    admin = session.query(User).filter(User.username == admin_username).first()
    
    if not admin:
        # Create admin user
        admin = User(
            id=str(uuid.uuid4()),
            username=admin_username,
            email=admin_email,
            password_hash=bcrypt.hash(admin_password),
            is_active=True,
            is_admin=True
        )
        session.add(admin)
        session.commit()
        print(f'Admin user created: {admin_username}')
    else:
        print(f'Admin user already exists: {admin_username}')
"

echo "Database initialization complete"
