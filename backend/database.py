import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Absolute path: store the DB alongside uploads/cams under runtime/
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
DB_PATH = _PROJECT_ROOT / "runtime" / "patient_records.db"

# Ensure the parent directory exists at import time (once).
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def initialize_database() -> None:
    """
    Initialize the SQLite database with the patient table.
    Creates the database and table if they don't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the patient table with all required fields
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            image_path TEXT,
            prediction TEXT,
            confidence REAL,
            date TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized at: {DB_PATH}")


def insert_patient_record(
    name: str,
    age: int,
    image_path: Optional[str] = None,
    prediction: Optional[str] = None,
    confidence: Optional[float] = None,
    date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Insert a patient record into the database.
    
    Args:
        name (str): Patient's name (required)
        age (int): Patient's age (required)
        image_path (str, optional): Path to the patient's image
        prediction (str, optional): Medical prediction or diagnosis
        confidence (float, optional): Confidence score (0.0 to 1.0)
        date (str, optional): Date of the record (ISO format). Defaults to current date.
    
    Returns:
        dict: Response containing:
            - success (bool): Whether insertion was successful
            - message (str): Success or error message
            - record_id (int): ID of the inserted record (if successful)
            - data (dict): The inserted record details
    
    Example:
        result = insert_patient_record(
            name="John Doe",
            age=45,
            image_path="/path/to/image.jpg",
            prediction="Pneumonia",
            confidence=0.92,
            date="2025-03-24"
        )
        print(result)
    """
    try:
        # Validate required fields
        if not name or not isinstance(name, str) or name.strip() == "":
            return {
                "success": False,
                "message": "Error: 'name' is required and must be a non-empty string"
            }
        
        if not isinstance(age, int) or age < 0 or age > 120:
            return {
                "success": False,
                "message": "Error: 'age' must be an integer between 0 and 150"
            }
        
        # Validate optional fields
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                return {
                    "success": False,
                    "message": "Error: 'confidence' must be a number between 0 and 1"
                }
        
        # Use current date if not provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {
                "success": False,
                "message": "Error: 'date' must be in ISO format (YYYY-MM-DD)"
            }
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert the record
        cursor.execute("""
            INSERT INTO patient (name, age, image_path, prediction, confidence, date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, age, image_path, prediction, confidence, date))
        
        # Get the ID of the inserted record
        record_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        # Return success response
        return {
            "success": True,
            "message": "Patient record inserted successfully",
            "record_id": record_id,
            "data": {
                "id": record_id,
                "name": name,
                "age": age,
                "image_path": image_path,
                "prediction": prediction,
                "confidence": confidence,
                "date": date
            }
        }
    
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }


def get_all_patients() -> Dict[str, Any]:
    """
    Retrieve all patient records from the database.
    
    Returns:
        dict: Response containing:
            - success (bool): Whether query was successful
            - count (int): Number of records returned
            - patients (list): List of all patient records
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM patient ORDER BY date DESC")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        patients = [dict(zip(columns, row)) for row in rows]
        
        return {
            "success": True,
            "count": len(patients),
            "patients": patients
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Error retrieving patients: {str(e)}"
        }


def get_patient_by_id(patient_id: int) -> Dict[str, Any]:
    """
    Retrieve a specific patient record by ID.
    
    Args:
        patient_id (int): The ID of the patient to retrieve
    
    Returns:
        dict: Response containing patient data or error message
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM patient WHERE id = ?", (patient_id,))
        columns = [description[0] for description in cursor.description]
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            patient = dict(zip(columns, row))
            return {
                "success": True,
                "patient": patient
            }
        else:
            return {
                "success": False,
                "message": f"No patient found with ID {patient_id}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Error retrieving patient: {str(e)}"
        }


def update_patient_record(
    patient_id: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Update a patient record.
    
    Args:
        patient_id (int): The ID of the patient to update
        **kwargs: Fields to update (name, age, image_path, prediction, confidence, date)
    
    Returns:
        dict: Response containing success status and updated data
    """
    try:
        allowed_fields = {"name", "age", "image_path", "prediction", "confidence", "date"}
        update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not update_fields:
            return {
                "success": False,
                "message": "No valid fields provided for update"
            }
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Build the UPDATE query dynamically
        set_clause = ", ".join([f"{field} = ?" for field in update_fields.keys()])
        values = list(update_fields.values()) + [patient_id]
        
        cursor.execute(f"UPDATE patient SET {set_clause} WHERE id = ?", values)
        conn.commit()
        
        if cursor.rowcount == 0:
            conn.close()
            return {
                "success": False,
                "message": f"No patient found with ID {patient_id}"
            }
        
        # Fetch the updated record
        cursor.execute("SELECT * FROM patient WHERE id = ?", (patient_id,))
        columns = [description[0] for description in cursor.description]
        row = cursor.fetchone()
        conn.close()
        
        patient = dict(zip(columns, row))
        
        return {
            "success": True,
            "message": "Patient record updated successfully",
            "patient": patient
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Error updating patient: {str(e)}"
        }
