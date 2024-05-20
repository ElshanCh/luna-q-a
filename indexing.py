import sys
import os
sys.path.append('../')
from utils.Indexer import *

# Index Document
pdf_relative_path = "Indexing\data\SP_1334.pdf"
pdf_absolute_path = os.path.join(dir_path, pdf_relative_path)
vector_store = VectorStore(pdf_absolute_path)
vector_store.index_document()