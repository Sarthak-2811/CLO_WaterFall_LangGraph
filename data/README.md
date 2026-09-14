# Data Directory

This folder contains the raw documents, processed vector embeddings, and local database files required for the application. 

> **⚠️ Important Git Note:** Do not commit the contents of this folder to version control. The files here are generated locally or contain raw data. Ensure your `.gitignore` includes `data/*` but excludes `data/README.md` (e.g., `!data/README.md`).

## Directory Structure

As referenced in `image_616641.png`, the `data` directory follows a specific hierarchical structure:

```text
data/
├── raw_indentures/          # Contains the original PDF documents
├── vector_store/            # Contains processed vector embeddings
├── README.md                # This file
├── workflow_threads.db      # SQLite database for workflow states
├── workflow_threads.db-shm  # SQLite shared memory file
└── workflow_threads.db-wal  # SQLite write-ahead log