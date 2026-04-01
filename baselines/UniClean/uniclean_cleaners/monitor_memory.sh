#!/bin/bash

# Process ID to monitor
PID=791193

# Memory limit in MB (800 GB = 819200 MB)
MEMORY_LIMIT_MB=819200

# Log file for memory monitoring
LOG_FILE="memory_monitor_${PID}.log"

echo "Starting memory monitor for process $PID with limit of 800 GB" | tee -a "$LOG_FILE"
echo "$(date): Monitoring started" | tee -a "$LOG_FILE"

# Check if process exists
if ! ps -p $PID > /dev/null 2>&1; then
    echo "ERROR: Process $PID not found!" | tee -a "$LOG_FILE"
    exit 1
fi

# Function to get memory usage in MB for a process and all its children
get_memory_usage() {
    local pid=$1
    local total_mem=0
    
    # Get memory for main process and all children
    if ps -p "$pid" > /dev/null 2>&1; then
        # Get memory in KB, then convert to MB
        local mem_kb=$(ps -o pid=,rss= -p "$pid" --ppid "$pid" 2>/dev/null | awk '{sum+=$2} END {print sum}')
        if [ -n "$mem_kb" ]; then
            total_mem=$((mem_kb / 1024))
        fi
    fi
    
    echo "$total_mem"
}

# Monitor the process
while ps -p $PID > /dev/null 2>&1; do
    # Get current memory usage
    CURRENT_MEM=$(get_memory_usage $PID)
    
    # Calculate percentage
    PERCENT=$((CURRENT_MEM * 100 / MEMORY_LIMIT_MB))
    
    # Log memory usage every check
    echo "$(date): Memory usage: ${CURRENT_MEM} MB / ${MEMORY_LIMIT_MB} MB (${PERCENT}%)" | tee -a "$LOG_FILE"
    
    # Check if memory limit exceeded
    if [ "$CURRENT_MEM" -gt "$MEMORY_LIMIT_MB" ]; then
        echo "$(date): *** ALERT! Memory limit exceeded: ${CURRENT_MEM} MB > ${MEMORY_LIMIT_MB} MB ***" | tee -a "$LOG_FILE"
        echo "$(date): Killing process $PID and all its children" | tee -a "$LOG_FILE"
        
        # Kill the process and all its children
        pkill -TERM -P $PID
        sleep 2
        
        # Force kill if still running
        if ps -p $PID > /dev/null 2>&1; then
            pkill -KILL -P $PID
            kill -KILL $PID 2>/dev/null
            echo "$(date): Process force killed" | tee -a "$LOG_FILE"
        else
            echo "$(date): Process terminated gracefully" | tee -a "$LOG_FILE"
        fi
        
        echo "$(date): Process terminated due to excessive memory usage" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    # Check every 30 seconds
    sleep 30
done

echo "$(date): Process $PID has completed or terminated" | tee -a "$LOG_FILE"
