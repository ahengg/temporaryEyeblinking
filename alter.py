import psycopg2

# Replace these values with your actual database connection details
db_params = {
    'host': 'localhost',
    'database': 'liveness_memo',
    'user': 'postgres',
    'password': 'test123',
    'port': '8000'
}

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(**db_params)

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Example query: add a new column 'new_column' to the 'liveness_score' table
query = "ALTER TABLE liveness_score ADD COLUMN file_name varchar(50);"

# Execute the query
cursor.execute(query)

# Commit the changes
conn.commit()

# Print a message indicating success
print("Column added successfully")

# Close the cursor and connection
cursor.close()
conn.close()
