import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()
BASE_DATA_DIR = os.getenv('BASE_DATA_DIR')
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR')
PREPROCESSED_DATA_DIR = os.getenv('PREPROCESSED_DATA_DIR')
MLOMIX_SAMPLESHEETS_DIR = os.getenv('MLOMIX_SAMPLESHEETS_DIR')

GEX_EXCLUDE_SUBTYPES = [
    'low hyperdiploid',
    'RUNX1::RUNX1T1-like',
    'IGH::CEBPD',
    'PICALM::MLLT10'
]
DNAM_EXCLUDE_SUBTYPES = [
    'low hyperdiploid',
    'MECOM-r',
    'hypodiploid'
]
