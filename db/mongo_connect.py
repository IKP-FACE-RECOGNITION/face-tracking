from pymongo import MongoClient

def initialize_mongo_connection(uri,db_name):
    # Define your MongoDB connection details
    mongo_uri = uri  
    database_name = db_name        
    
    # Create a MongoDB client
    client = MongoClient(mongo_uri)

    # Connect to the specific database
    database = client[database_name]

    print("Connected to MongoDB!")
    return database