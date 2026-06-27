"""
Eval test set — 18 cases, 6 per route.

Each case:
  id               — unique identifier
  question         — sent to the graph exactly as-is
  expected_route   — "rag" | "sql" | "tool"
  expected_keywords — substrings (case-insensitive) that must appear in the answer.
                      Empty list = routing-only check; keyword score is skipped.
  description      — human-readable label for the scorecard

SQL expected values are derived from the seed data in scripts/seed_db.py:
  - delivered: 8 shipments   (SHP-001,003,005,007,009,011,013,015)
  - in_transit: 4 shipments  (SHP-002,006,010,014)
  - pending: 2 shipments     (SHP-004,SHP-012)
  - cancelled: 1 shipment    (SHP-008)
  - total: 15 shipments
  - heaviest: SHP-015, 14.0 kg
  - average weight: ~6.09 kg

RAG expected_keywords are empty because the PDF content is user-supplied and
unknown at test-set authoring time. Routing accuracy is still validated.
"""

TEST_CASES: list[dict] = [
    # ------------------------------------------------------------------
    # RAG — 6 cases (routing only)
    # ------------------------------------------------------------------
    {
        "id": "rag_01",
        "question": "What is this document about?",
        "expected_route": "rag",
        "expected_keywords": [],
        "description": "generic content query",
    },
    {
        "id": "rag_02",
        "question": "Summarize the main points of the document.",
        "expected_route": "rag",
        "expected_keywords": [],
        "description": "summarisation request",
    },
    {
        "id": "rag_03",
        "question": "According to the document, what skills are mentioned?",
        "expected_route": "rag",
        "expected_keywords": [],
        "description": "'according to' trigger phrase",
    },
    {
        "id": "rag_04",
        "question": "What does the PDF say about experience?",
        "expected_route": "rag",
        "expected_keywords": [],
        "description": "explicit PDF reference",
    },
    {
        "id": "rag_05",
        "question": "What qualifications are listed in the document?",
        "expected_route": "rag",
        "expected_keywords": [],
        "description": "qualifications query",
    },
    {
        "id": "rag_06",
        "question": "Can you find any contact information in the uploaded file?",
        "expected_route": "rag",
        "expected_keywords": [],
        "description": "contact info from uploaded file",
    },

    # ------------------------------------------------------------------
    # SQL — 6 cases (with expected keywords from known seed data)
    # ------------------------------------------------------------------
    {
        "id": "sql_01",
        "question": "How many shipments have been delivered?",
        "expected_route": "sql",
        "expected_keywords": ["8"],
        "description": "count by status=delivered",
    },
    {
        "id": "sql_02",
        "question": "How many shipments are currently in transit?",
        "expected_route": "sql",
        "expected_keywords": ["4"],
        "description": "count by status=in_transit",
    },
    {
        "id": "sql_03",
        "question": "What is the total number of shipments in the database?",
        "expected_route": "sql",
        "expected_keywords": ["15"],
        "description": "total row count",
    },
    {
        "id": "sql_04",
        "question": "Which shipment has the highest weight?",
        "expected_route": "sql",
        "expected_keywords": ["SHP-015", "14"],
        "description": "max aggregate query",
    },
    {
        "id": "sql_05",
        "question": "List all pending shipments.",
        "expected_route": "sql",
        "expected_keywords": ["SHP-004"],
        "description": "filter by status=pending",
    },
    {
        "id": "sql_06",
        "question": "What is the average weight of all shipments?",
        "expected_route": "sql",
        "expected_keywords": ["6"],
        "description": "avg aggregation (~6.09 kg)",
    },

    # ------------------------------------------------------------------
    # Tool — 6 cases: 3 file-listing, 3 calculator
    # ------------------------------------------------------------------
    {
        "id": "tool_01",
        "question": "What files are in the data directory?",
        "expected_route": "tool",
        "expected_keywords": ["resume", "shipments"],
        "description": "list data/ directory",
    },
    {
        "id": "tool_02",
        "question": "List the contents of the data folder.",
        "expected_route": "tool",
        "expected_keywords": ["shipments.db"],
        "description": "list data/ — alternate phrasing",
    },
    {
        "id": "tool_03",
        "question": "What files exist in the data directory?",
        "expected_route": "tool",
        "expected_keywords": ["resume"],
        "description": "existence check phrasing",
    },
    {
        "id": "tool_04",
        "question": "What is 25 times 4?",
        "expected_route": "tool",
        "expected_keywords": ["100"],
        "description": "simple multiplication",
    },
    {
        "id": "tool_05",
        "question": "Calculate the square root of 144.",
        "expected_route": "tool",
        "expected_keywords": ["12"],
        "description": "sqrt calculation",
    },
    {
        "id": "tool_06",
        "question": "What is 2 raised to the power of 10?",
        "expected_route": "tool",
        "expected_keywords": ["1024"],
        "description": "exponentiation",
    },
]
