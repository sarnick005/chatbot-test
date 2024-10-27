import sqlite3


def list_tables(database_name):
    """List tables in the SQLite database."""
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()
    if tables:
        print("Tables in database:")
        table_dict = {index + 1: table[0] for index, table in enumerate(tables)}
        for index, table_name in table_dict.items():
            print(f"{index}. {table_name}")
        return table_dict
    else:
        print("No tables found in database.")
        return {}


def display_table_contents(database_name, table_name):
    """Display contents of a specific table."""
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    if rows:
        print(f"\nContents of table '{table_name}':")
        for row in rows:
            print(row)
    else:
        print(f"\nTable '{table_name}' is empty.")

    conn.close()


def main():
    database_name = "chatbot.db"
    table_dict = list_tables(database_name)

    if table_dict:
        try:
            choice = int(
                input("Enter the number of the table you want to view, or 0 to exit: ")
            )
            if choice in table_dict:
                display_table_contents(database_name, table_dict[choice])
            elif choice == 0:
                print("Exiting.")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")


if __name__ == "__main__":
    main()
