"""SQLite e-commerce sample DB: schema, deterministic seed data, helpers."""

import random
import sqlite3

import pandas as pd

SCHEMA = """
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    price REAL NOT NULL,
    stock INTEGER NOT NULL
);
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    country TEXT NOT NULL,
    signup_date TEXT NOT NULL  -- YYYY-MM-DD
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,  -- YYYY-MM-DD
    status TEXT NOT NULL       -- pending | shipped | delivered | cancelled
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    rating INTEGER NOT NULL,   -- 1..5
    review_date TEXT NOT NULL
);
"""

CATEGORIES = ["Electronics", "Books", "Home", "Toys", "Sports", "Clothing", "Garden", "Beauty"]

# (name, category, price, stock)
PRODUCTS = [
    ("Wireless Mouse", "Electronics", 24.99, 120), ("Mechanical Keyboard", "Electronics", 89.0, 40),
    ("USB-C Hub", "Electronics", 39.5, 0), ("Noise Cancelling Headphones", "Electronics", 199.0, 15),
    ("4K Monitor", "Electronics", 329.0, 8), ("Smart Speaker", "Electronics", 59.0, 60),
    ("Python Crash Course", "Books", 31.0, 200), ("Designing Data-Intensive Apps", "Books", 45.0, 75),
    ("The Pragmatic Programmer", "Books", 38.5, 0), ("Clean Code", "Books", 33.0, 90),
    ("Cast Iron Skillet", "Home", 42.0, 30), ("French Press", "Home", 27.5, 55),
    ("LED Desk Lamp", "Home", 34.0, 70), ("Robot Vacuum", "Home", 249.0, 12),
    ("Chef Knife", "Home", 65.0, 25), ("Building Blocks Set", "Toys", 49.0, 80),
    ("Puzzle 1000pc", "Toys", 18.0, 110), ("RC Car", "Toys", 74.0, 0),
    ("Plush Bear", "Toys", 15.5, 140), ("Yoga Mat", "Sports", 22.0, 95),
    ("Dumbbell Set", "Sports", 120.0, 20), ("Running Shoes", "Sports", 95.0, 45),
    ("Cycling Helmet", "Sports", 58.0, 35), ("Jump Rope", "Sports", 9.99, 300),
    ("Denim Jacket", "Clothing", 79.0, 50), ("Wool Sweater", "Clothing", 64.0, 40),
    ("Cotton T-Shirt", "Clothing", 14.0, 500), ("Rain Coat", "Clothing", 88.0, 0),
    ("Garden Hose", "Garden", 29.0, 65), ("Pruning Shears", "Garden", 21.0, 85),
    ("Tomato Seeds", "Garden", 3.5, 400), ("Planter Box", "Garden", 46.0, 30),
    ("Face Serum", "Beauty", 36.0, 120), ("Shampoo", "Beauty", 11.0, 250),
    ("Electric Toothbrush", "Beauty", 69.0, 40), ("Sunscreen SPF50", "Beauty", 17.0, 180),
    ("Bluetooth Tracker", "Electronics", 29.0, 150), ("Webcam 1080p", "Electronics", 54.0, 33),
    ("Trail Backpack", "Sports", 110.0, 18), ("Board Game Night", "Toys", 39.0, 60),
]

COUNTRIES = ["USA", "Germany", "France", "India", "Brazil", "Japan"]
FIRST = ["Alice", "Bruno", "Chen", "Dana", "Emeka", "Farah", "Goran", "Hana", "Ivan", "Jun", "Kaia", "Liam"]
LAST = ["Novak", "Okafor", "Silva", "Tanaka", "Mehta"]
CUSTOMER_NAMES = [f"{FIRST[i % 12]} {LAST[i // 12]}" for i in range(60)]
STATUSES = ["delivered"] * 11 + ["shipped"] * 4 + ["pending"] * 3 + ["cancelled"] * 2


def _date(rng, start_year, end_year):
    return f"{rng.randint(start_year, end_year):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def build_db(path=":memory:", seed=7):
    """Create schema and seed deterministic data. Returns an open connection."""
    rng = random.Random(seed)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    cat_id = {c: i + 1 for i, c in enumerate(CATEGORIES)}
    conn.executemany("INSERT INTO categories VALUES (?, ?)", [(i, c) for c, i in cat_id.items()])
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
        [(i + 1, n, cat_id[c], p, s) for i, (n, c, p, s) in enumerate(PRODUCTS)],
    )
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        [
            # min(i % 8, 5) skews the split so "country with most customers" has a unique answer.
            (i + 1, n, n.lower().replace(" ", ".") + "@example.com", COUNTRIES[min(i % 8, 5)], _date(rng, 2021, 2023))
            for i, n in enumerate(CUSTOMER_NAMES)
        ],
    )
    # Customers 51..60 never order, so "customers without orders" is non-empty.
    orders = [(i + 1, rng.randint(1, 50), _date(rng, 2023, 2024), rng.choice(STATUSES)) for i in range(300)]
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
    items, item_id = [], 1
    for order_id, *_ in orders:
        for pid in rng.sample(range(1, len(PRODUCTS) + 1), rng.randint(1, 4)):
            items.append((item_id, order_id, pid, rng.randint(1, 5), PRODUCTS[pid - 1][2]))
            item_id += 1
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", items)
    pairs = set()
    while len(pairs) < 150:
        pairs.add((rng.randint(1, len(PRODUCTS)), rng.randint(1, 60)))
    reviews = [
        (i + 1, pid, cid, rng.choice([1, 2, 3, 3, 4, 4, 4, 5, 5, 5]), _date(rng, 2023, 2024))
        for i, (pid, cid) in enumerate(sorted(pairs))
    ]
    conn.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?)", reviews)
    conn.commit()
    return conn


def table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [r[0] for r in rows]


def schema_text(conn, tables=None):
    """DDL for the given tables (all if None), used as LM context."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return "\n".join(sql + ";" for name, sql in rows if tables is None or name in tables)


def run_sql(conn, sql):
    """Execute read-only SQL, return a DataFrame. Raises sqlite3.Error on bad SQL."""
    if not sql.strip().lower().startswith(("select", "with")):
        raise sqlite3.OperationalError("only SELECT statements are allowed")
    try:
        return pd.read_sql_query(sql, conn)
    except pd.errors.DatabaseError as e:
        raise e.__cause__ or sqlite3.OperationalError(str(e))  # unwrap to the short sqlite message
