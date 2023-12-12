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

# Example query: select all rows from a table named 'example_table'
query = "SELECT * FROM liveness_score;"

# Execute the query
cursor.execute(query)

# Fetch all the results
results = cursor.fetchall()

# Print the results
for row in results:
    print(row)
print("berhasi conmenct")
# Close the cursor and connection
cursor.close()
conn.close()
