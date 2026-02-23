import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()
BASE_DATA_DIR = os.getenv('BASE_DATA_DIR')
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR')
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR')
GEX_EXCLUDE_SUBTYPES = [
    'RUNX1::RUNX1T1-like',
    'low hyperdiploid',
    'IGH::CEBPD',
    'PICALM::MLLT10',
    'BCR::ABL1 AML'
]
DNAM_EXCLUDE_SUBTYPES = [
    'low hyperdiploid',
    'KAT6A-r',
    'RBM15::MKL1',
    'NUTM1-r'
]
