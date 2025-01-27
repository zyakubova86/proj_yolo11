import os

import psycopg2
from dotenv import load_dotenv
load_dotenv()

DATABASE_NAME = os.getenv("DB_NAME")
DATABASE_USER = os.getenv("DB_USER")
DATABASE_PASSWORD = os.getenv("PASSWORD")
DATABASE_HOST = os.getenv("HOST")
DATABASE_PORT = os.getenv("PORT")


def connect_db(db_name, db_user, password, host, port):
    """Connect to the database."""

    try:
        connection = psycopg2.connect(
            dbname=db_name, user=db_user, password=password, host=host, port=port
        )
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return

    # print("Connected to the database successfully!")
    return connection


def get_cameras_data():
    """Fetch RTSP links and camera IDs from the database."""
    connection = connect_db(DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, DATABASE_PORT)
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM carpets_camera ORDER BY id ASC;""")
    cameras = cursor.fetchall()

    cursor.close()
    connection.close()
    return cameras


def save_detection_to_db(counted_track_id, class_name, camera_id):
    """Save detection results to the database."""
    import time
    from datetime import datetime

    connection = connect_db(DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, DATABASE_PORT)
    cursor = connection.cursor()

    ts = time.time()
    detected_at = datetime.fromtimestamp(ts).strftime("%d-%m-%Y %H:%M:%S")

    cursor.execute("""
            INSERT INTO carpets_carpet (carpet_track_id, classname, created_at, camera_id_id) VALUES (%s, %s, %s, %s);""",
        (counted_track_id, class_name, detected_at, camera_id),
    )
    connection.commit()

    count_row = cursor.rowcount

    print(f"camera_id: {camera_id}, cls:{class_name}, time: {detected_at}, track_id: {counted_track_id} inserted db")
    print("-----------------------------------------------------------------------------------------------------------")

    # cursor.close()
    # connection.close()


