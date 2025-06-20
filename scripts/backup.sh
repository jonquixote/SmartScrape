#!/bin/bash
# backup.sh - Automated backup script for SmartScrape
# Usage: ./backup.sh [backup_directory]

set -e

# Default backup directory is ./backups
BACKUP_DIR=${1:-"./backups"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="smartscrape_backup_$TIMESTAMP"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

# Create the backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting SmartScrape backup process..."
log "Backup will be saved to: $BACKUP_PATH"

# Create backup directory structure
mkdir -p "$BACKUP_PATH/config"
mkdir -p "$BACKUP_PATH/data"
mkdir -p "$BACKUP_PATH/logs"

# Backup configuration files
log "Backing up configuration files..."
cp -r .env* "$BACKUP_PATH/config/" 2>/dev/null || log "No .env files found"
cp -r docker/ "$BACKUP_PATH/config/" 2>/dev/null || log "No docker configuration found"
cp -r monitoring/ "$BACKUP_PATH/config/" 2>/dev/null || log "No monitoring configuration found"

# Backup data files
log "Backing up data files..."
cp -r extraction_strategies/ "$BACKUP_PATH/data/" 2>/dev/null || log "No extraction strategies found"
cp -r cache/ "$BACKUP_PATH/data/" 2>/dev/null || log "No cache files found"
cp -r data/ "$BACKUP_PATH/data/" 2>/dev/null || log "No data files found"

# Backup logs (last 7 days)
log "Backing up recent logs..."
find ./logs -type f -name "*.log" -mtime -7 -exec cp {} "$BACKUP_PATH/logs/" \; 2>/dev/null || log "No log files found"

# Backup database if available
if [ -f ".env.production" ] && grep -q "DATABASE_URL" .env.production; then
    log "Database configuration found, attempting database backup..."
    source .env.production
    
    # Extract database type and connection info from DATABASE_URL
    DB_TYPE=$(echo $DATABASE_URL | cut -d':' -f1)
    
    if [ "$DB_TYPE" = "postgresql" ]; then
        log "Backing up PostgreSQL database..."
        # Extract connection details from URL
        DB_USER=$(echo $DATABASE_URL | sed -e 's/.*:\/\/\(.*\):.*/\1/')
        DB_PASSWORD=$(echo $DATABASE_URL | sed -e 's/.*:\/\/.*:\(.*\)@.*/\1/')
        DB_HOST=$(echo $DATABASE_URL | sed -e 's/.*@\(.*\)\/.*/\1/')
        DB_NAME=$(echo $DATABASE_URL | sed -e 's/.*\/\(.*\)/\1/')
        
        # Create database backup
        mkdir -p "$BACKUP_PATH/database"
        export PGPASSWORD=$DB_PASSWORD
        pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME -F c -b -v -f "$BACKUP_PATH/database/$DB_NAME.dump" || log "Database backup failed"
    elif [ "$DB_TYPE" = "mysql" ]; then
        log "Backing up MySQL database..."
        # Extract connection details from URL
        DB_USER=$(echo $DATABASE_URL | sed -e 's/.*:\/\/\(.*\):.*/\1/')
        DB_PASSWORD=$(echo $DATABASE_URL | sed -e 's/.*:\/\/.*:\(.*\)@.*/\1/')
        DB_HOST=$(echo $DATABASE_URL | sed -e 's/.*@\(.*\)\/.*/\1/')
        DB_NAME=$(echo $DATABASE_URL | sed -e 's/.*\/\(.*\)/\1/')
        
        # Create database backup
        mkdir -p "$BACKUP_PATH/database"
        mysqldump -h $DB_HOST -u $DB_USER -p$DB_PASSWORD $DB_NAME > "$BACKUP_PATH/database/$DB_NAME.sql" || log "Database backup failed"
    elif [ "$DB_TYPE" = "sqlite" ]; then
        log "Backing up SQLite database..."
        DB_PATH=$(echo $DATABASE_URL | sed -e 's/sqlite:\/\/\///')
        
        # Create database backup
        mkdir -p "$BACKUP_PATH/database"
        cp $DB_PATH "$BACKUP_PATH/database/" || log "SQLite backup failed"
    else
        log "Unknown database type, skipping database backup"
    fi
else
    log "No database configuration found, skipping database backup"
fi

# Create a tarball of the backup
log "Creating compressed backup archive..."
tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME"

# Clean up the uncompressed backup directory
log "Cleaning up temporary files..."
rm -rf "$BACKUP_PATH"

# Calculate backup size
BACKUP_SIZE=$(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)

# Cleanup old backups (older than 30 days) if enabled
if [ -n "$BACKUP_RETENTION_DAYS" ] && [ "$BACKUP_RETENTION_DAYS" -gt 0 ]; then
    log "Cleaning up backups older than $BACKUP_RETENTION_DAYS days..."
    find "$BACKUP_DIR" -name "smartscrape_backup_*.tar.gz" -type f -mtime +$BACKUP_RETENTION_DAYS -delete
fi

log "Backup completed successfully: ${BACKUP_NAME}.tar.gz (${BACKUP_SIZE})"
log "Backup stored in $BACKUP_DIR"
