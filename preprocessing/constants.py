import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()
BASE_DATA_DIR = os.getenv('BASE_DATA_DIR')
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR')
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR')
