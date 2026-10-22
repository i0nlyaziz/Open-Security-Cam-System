import sqlite3

def create():
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()
    cursor.execute("""
        create table if not exists user (
            id integer primary key, 
            date text, 
            conf real
        )
    """)
    conn.commit()
    conn.close()

def fill(date, conf):
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()
    cursor.execute("insert into user(date, conf) values(?, ?)", (date, conf))
    conn.commit()
    conn.close()