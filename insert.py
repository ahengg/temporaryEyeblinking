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

# Example query: insert data into the 'liveness_score' table
query = "INSERT INTO liveness_score (nip, total_blink, tanggal_absen, liveness_passive, pass) VALUES (23020235, 20, '2023-10-10', 98, 1);"

# Execute the query
cursor.execute(query)

# Commit the changes
conn.commit()

# Print a message indicating success
print("Data inserted successfully")

# Close the cursor and connection
cursor.close()
conn.close()
