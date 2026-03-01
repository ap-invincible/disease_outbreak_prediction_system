import sqlite3

def migrate():
    try:
        conn = sqlite3.connect('outbreak.db')
        c = conn.cursor()
        c.execute("ALTER TABLE user ADD COLUMN email VARCHAR(120)")
        conn.commit()
        print("Successfully added email column to user table")
    except Exception as e:
        print(f"Migration error (already migrated?): {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    migrate()
