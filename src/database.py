"""
Database module for Face Authentication Attendance System
Handles user registration and attendance records using SQLite
"""

import json
import sqlite3
from datetime import datetime, date
from typing import Optional, List, Tuple
import os
import numpy as np

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "attendance.db")


def get_connection():
    """Get database connection"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Users table - stores face embeddings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            email TEXT,
            embedding TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Attendance table - stores punch records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            punch_type TEXT NOT NULL CHECK(punch_type IN ('IN', 'OUT')),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    conn.close()


def register_user(name: str, embedding: np.ndarray, email: str = None) -> Tuple[bool, str]:
    """
    Register a new user with their face embedding
    
    Args:
        name: User's name (unique identifier)
        embedding: Face embedding vector (512D for ArcFace)
        email: Optional email address
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Convert embedding to JSON string for storage
        embedding_json = json.dumps(embedding.tolist())
        
        cursor.execute(
            "INSERT INTO users (name, email, embedding) VALUES (?, ?, ?)",
            (name, email, embedding_json)
        )
        conn.commit()
        return True, f"User '{name}' registered successfully!"
    
    except sqlite3.IntegrityError:
        return False, f"User '{name}' already exists!"
    
    except Exception as e:
        return False, f"Registration failed: {str(e)}"
    
    finally:
        conn.close()


def get_all_users() -> List[dict]:
    """Get all registered users with their embeddings"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, email, embedding, created_at FROM users")
    rows = cursor.fetchall()
    conn.close()
    
    users = []
    for row in rows:
        users.append({
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "embedding": np.array(json.loads(row["embedding"])),
            "created_at": row["created_at"]
        })
    
    return users


def get_user_by_name(name: str) -> Optional[dict]:
    """Get user by name"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "embedding": np.array(json.loads(row["embedding"])),
            "created_at": row["created_at"]
        }
    return None


def delete_user(name: str) -> Tuple[bool, str]:
    """Delete a user by name"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM users WHERE name = ?", (name,))
        if cursor.rowcount > 0:
            conn.commit()
            return True, f"User '{name}' deleted successfully!"
        return False, f"User '{name}' not found!"
    except Exception as e:
        return False, f"Delete failed: {str(e)}"
    finally:
        conn.close()


def record_attendance(user_id: int, punch_type: str, confidence: float = None) -> Tuple[bool, str]:
    """
    Record attendance (punch in/out)
    
    Args:
        user_id: User's database ID
        punch_type: 'IN' or 'OUT'
        confidence: Face match confidence score
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO attendance (user_id, punch_type, confidence) VALUES (?, ?, ?)",
            (user_id, punch_type, confidence)
        )
        conn.commit()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True, f"Punch {punch_type} recorded at {timestamp}"
    
    except Exception as e:
        return False, f"Failed to record attendance: {str(e)}"
    
    finally:
        conn.close()


def get_last_punch(user_id: int, today_only: bool = True) -> Optional[str]:
    """
    Get the last punch type for a user
    
    Args:
        user_id: User's database ID
        today_only: If True, only check today's records
    
    Returns:
        'IN', 'OUT', or None if no records
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if today_only:
        today = date.today().isoformat()
        cursor.execute("""
            SELECT punch_type FROM attendance 
            WHERE user_id = ? AND DATE(timestamp) = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (user_id, today))
    else:
        cursor.execute("""
            SELECT punch_type FROM attendance 
            WHERE user_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (user_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    return row["punch_type"] if row else None


def get_attendance_history(user_id: int = None, limit: int = 50) -> List[dict]:
    """
    Get attendance history
    
    Args:
        user_id: Optional user ID to filter by
        limit: Maximum number of records to return
    
    Returns:
        List of attendance records
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute("""
            SELECT a.id, u.name, a.punch_type, a.timestamp, a.confidence
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE a.user_id = ?
            ORDER BY a.timestamp DESC
            LIMIT ?
        """, (user_id, limit))
    else:
        cursor.execute("""
            SELECT a.id, u.name, a.punch_type, a.timestamp, a.confidence
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            ORDER BY a.timestamp DESC
            LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_today_attendance() -> List[dict]:
    """Get all attendance records for today"""
    conn = get_connection()
    cursor = conn.cursor()
    
    today = date.today().isoformat()
    cursor.execute("""
        SELECT a.id, u.name, a.punch_type, a.timestamp, a.confidence
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE DATE(a.timestamp) = ?
        ORDER BY a.timestamp DESC
    """, (today,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


# Initialize database on module import
init_db()
