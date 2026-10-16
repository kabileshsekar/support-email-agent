"""Create the sample Device & Order database (runtime/support.db).

All dates are stored relative to the day the seed runs, so order ETAs and warranty
states stay realistic whenever the project is run.

    python -m data.seed
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from agent.config import DB_PATH

TODAY = date.today()


def d(offset_days: int) -> str:
    return (TODAY + timedelta(days=offset_days)).isoformat()


PRODUCTS = [
    # sku, name, category, price, warranty_months, latest_firmware, specs
    ("NB-SPK-200", "Nimbus Echo Speaker 200", "Smart speaker", 129.00, 12, "3.4.0", {
        "drivers": "2 x 2-inch full-range + 1 x 3.5-inch woofer", "connectivity": "Wi-Fi 5 (2.4/5 GHz), Bluetooth 5.2",
        "voice_assistant": "Nimbus Voice, Alexa built-in", "power": "AC adapter, 24 W", "dimensions": "150 x 150 x 180 mm",
        "weight": "1.1 kg", "colours": "Charcoal, Chalk", "multiroom": "Yes, up to 8 speakers"}),
    ("NB-BUD-PRO", "Nimbus AirBuds Pro", "Wireless earbuds", 179.00, 12, "1.8.2", {
        "battery": "8 h per charge (ANC on: 6 h), 32 h total with case", "charging": "USB-C and Qi wireless; 10 min = 1.5 h",
        "noise_cancelling": "Hybrid ANC with transparency mode", "water_resistance": "IPX4 (sweat and splash)",
        "bluetooth": "5.3, multipoint (2 devices)", "codecs": "SBC, AAC, LDAC", "weight": "5.2 g per bud"}),
    ("NB-WCH-3", "Nimbus Pulse Watch 3", "Smartwatch", 249.00, 12, "2.1.0", {
        "display": "1.4-inch AMOLED, always-on option", "battery": "Up to 5 days typical, 30 h with GPS",
        "water_resistance": "5 ATM (swimming, not diving)", "sensors": "Heart rate, SpO2, accelerometer, GPS, barometer",
        "compatibility": "Android 9+ and iOS 15+ via Nimbus app", "charging": "Magnetic puck, 0-100% in 75 min"}),
    ("NB-RTR-6", "Nimbus Mesh Router AX6", "Wi-Fi router", 199.00, 24, "5.0.3", {
        "wifi_standard": "Wi-Fi 6 (802.11ax), dual band", "speed": "Up to 3000 Mbps combined",
        "coverage": "Up to 2,500 sq ft per node", "ports": "4 x Gigabit Ethernet (1 WAN, 3 LAN)",
        "mesh": "Add up to 6 nodes", "security": "WPA3, automatic security updates"}),
    ("NB-VAC-X1", "Nimbus RoboClean X1", "Robot vacuum", 399.00, 24, "4.2.0", {
        "suction": "4000 Pa", "battery": "Up to 180 min on eco mode", "navigation": "LiDAR mapping, multi-floor maps",
        "dustbin": "0.4 L", "mopping": "Yes, 250 ml water tank", "noise": "62 dB on standard mode",
        "auto_empty": "Compatible with X1 Auto-Empty Dock (sold separately)"}),
    ("NB-TAB-10", "Nimbus Slate 10", "Tablet", 329.00, 12, "14.2", {
        "display": "10.5-inch 2K IPS, 90 Hz", "memory": "8 GB RAM / 128 GB storage, microSD up to 1 TB",
        "battery": "Up to 12 h video playback", "os": "Android 14", "cameras": "13 MP rear, 8 MP front",
        "weight": "480 g"}),
    ("NB-CAM-2K", "Nimbus Guard Cam 2K", "Security camera", 89.00, 12, "2.6.1", {
        "resolution": "2K (2560 x 1440)", "night_vision": "Infrared up to 10 m, colour night mode",
        "weatherproof": "IP65 (indoor/outdoor)", "storage": "microSD up to 256 GB or Nimbus Cloud",
        "connectivity": "2.4 GHz Wi-Fi only", "audio": "Two-way talk"}),
]

CUSTOMERS = [
    ("C001", "Priya Sharma", "priya.sharma@example.com", "+1-415-555-0101", "gold", d(-900)),
    ("C002", "Daniel Okafor", "daniel.okafor@example.com", "+1-312-555-0102", "standard", d(-400)),
    ("C003", "Mei Lin Chen", "meilin.chen@example.com", "+1-206-555-0103", "standard", d(-30)),
    ("C004", "Lucas Fernandes", "lucas.fernandes@example.com", "+1-617-555-0104", "standard", d(-800)),
    ("C005", "Aisha Rahman", "aisha.rahman@example.com", "+1-646-555-0105", "gold", d(-600)),
    ("C006", "Tom Becker", "tom.becker@example.com", "+1-503-555-0106", "standard", d(-520)),
    ("C007", "Sofia Rossi", "sofia.rossi@example.com", "+1-305-555-0107", "standard", d(-60)),
    ("C008", "Arjun Mehta", "arjun.mehta@example.com", "+1-408-555-0108", "standard", d(-100)),
    ("C009", "Emma Wilson", "emma.wilson@example.com", "+1-720-555-0109", "platinum", d(-1000)),
    ("C010", "Kenji Watanabe", "kenji.watanabe@example.com", "+1-213-555-0110", "standard", d(-40)),
    ("C011", "Fatima Zahra", "fatima.zahra@example.com", "+1-202-555-0111", "standard", d(-15)),
    ("C012", "Noah Kim", "noah.kim@example.com", "+1-512-555-0112", "gold", d(-330)),
]

# order_id, customer, placed, status, carrier, tracking, eta, shipping_method, note, items[(sku, qty)]
ORDERS = [
    ("ORD-10421", "C001", d(-3), "shipped", "UPS", "1Z999AA10123456784", d(2), "standard", None, [("NB-BUD-PRO", 1)]),
    ("ORD-10422", "C002", d(-1), "processing", None, None, d(5), "standard", None, [("NB-VAC-X1", 1)]),
    ("ORD-10423", "C003", d(-12), "delivered", "UPS", "1Z999AA10123456791", d(-7), "standard", None, [("NB-WCH-3", 1)]),
    ("ORD-10424", "C004", d(-9), "delayed", "FedEx", "784512369874", d(4), "standard",
     "Weather delay at the regional hub; new estimated delivery date provided by carrier.", [("NB-RTR-6", 2)]),
    ("ORD-10425", "C005", d(-20), "delivered", "DHL", "JD014600006281234567", d(-16), "express", None,
     [("NB-TAB-10", 1), ("NB-CAM-2K", 1)]),
    ("ORD-10426", "C006", d(-2), "out_for_delivery", "UPS", "1Z999AA10123456808", d(0), "express", None, [("NB-SPK-200", 1)]),
    ("ORD-10427", "C007", d(-6), "cancelled", None, None, None, "standard",
     "Cancelled by customer before shipping. Refund of $249.00 issued to original payment method.", [("NB-WCH-3", 1)]),
    ("ORD-10428", "C001", d(-40), "delivered", "FedEx", "784512369111", d(-36), "standard", None, [("NB-SPK-200", 1)]),
    ("ORD-10429", "C008", d(-15), "returned", "UPS", "1Z999AA10123456815", d(-11), "standard",
     "Return received at warehouse; refund of $179.00 is being processed (5-7 business days).", [("NB-BUD-PRO", 1)]),
    ("ORD-10430", "C009", d(-4), "shipped", "FedEx", "784512369555", d(1), "standard", None, [("NB-CAM-2K", 3)]),
    ("ORD-10431", "C010", d(-25), "delivered", "DHL", "JD014600006281239999", d(-21), "standard", None, [("NB-VAC-X1", 1)]),
    ("ORD-10432", "C011", d(-8), "delivered", "UPS", "1Z999AA10123456822", d(-4), "standard", None, [("NB-RTR-6", 1)]),
    ("ORD-10433", "C012", d(-1), "processing", None, None, d(4), "standard", None, [("NB-TAB-10", 1)]),
    ("ORD-10434", "C002", d(-200), "delivered", "UPS", "1Z999AA10123450001", d(-196), "standard", None, [("NB-WCH-3", 1)]),
]

# serial, sku, customer, purchase_offset_days, firmware, order_id, plan
DEVICES = [
    ("SN-SPK-50121", "NB-SPK-200", "C001", -36, "3.2.1", "ORD-10428", "standard"),
    ("SN-WCH-30455", "NB-WCH-3", "C003", -7, "2.0.4", "ORD-10423", "standard"),
    ("SN-TAB-61002", "NB-TAB-10", "C005", -16, "14.2", "ORD-10425", "standard"),
    ("SN-CAM-88140", "NB-CAM-2K", "C005", -16, "2.5.0", "ORD-10425", "standard"),
    ("SN-VAC-90017", "NB-VAC-X1", "C010", -21, "4.1.3", "ORD-10431", "standard"),
    ("SN-RTR-44210", "NB-RTR-6", "C011", -4, "5.0.3", "ORD-10432", "standard"),
    ("SN-WCH-29873", "NB-WCH-3", "C002", -196, "2.1.0", "ORD-10434", "standard"),
    ("SN-SPK-40877", "NB-SPK-200", "C006", -500, "3.1.0", None, "standard"),      # warranty expired
    ("SN-VAC-81234", "NB-VAC-X1", "C009", -800, "4.2.0", None, "nimbuscare_plus"),  # extended, still active
    ("SN-RTR-31120", "NB-RTR-6", "C004", -760, "4.9.1", None, "standard"),        # expired (24 m)
    ("SN-BUD-52001", "NB-BUD-PRO", "C012", -300, "1.7.0", None, "standard"),
]
EXTENDED_MONTHS = 24
CLAIMS = {"SN-BUD-52001": 1}

SCHEMA = """
CREATE TABLE customers (customer_id TEXT PRIMARY KEY, name TEXT, email TEXT UNIQUE, phone TEXT,
                        tier TEXT, member_since TEXT);
CREATE TABLE products  (sku TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL,
                        warranty_months INTEGER, latest_firmware TEXT, specs TEXT);
CREATE TABLE orders    (order_id TEXT PRIMARY KEY, customer_id TEXT REFERENCES customers, placed_date TEXT,
                        status TEXT, carrier TEXT, tracking_number TEXT, estimated_delivery TEXT,
                        shipping_method TEXT, note TEXT, total REAL);
CREATE TABLE order_items (order_id TEXT REFERENCES orders, sku TEXT REFERENCES products, quantity INTEGER,
                          unit_price REAL);
CREATE TABLE devices   (serial_number TEXT PRIMARY KEY, sku TEXT REFERENCES products,
                        customer_id TEXT REFERENCES customers, purchase_date TEXT, firmware_version TEXT,
                        order_id TEXT);
CREATE TABLE warranties (serial_number TEXT PRIMARY KEY REFERENCES devices, plan TEXT, start_date TEXT,
                         end_date TEXT, claims_used INTEGER, coverage TEXT);
"""


def add_months(day: date, months: int) -> date:
    y, m = divmod(day.month - 1 + months, 12)
    year, month = day.year + y, m + 1
    for dd in (day.day, 30, 29, 28):
        try:
            return date(year, month, dd)
        except ValueError:
            continue
    raise ValueError


def seed(path: Path = DB_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", CUSTOMERS)
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)",
                    [(s, n, c, p, w, f, json.dumps(sp)) for s, n, c, p, w, f, sp in PRODUCTS])
    price = {p[0]: p[3] for p in PRODUCTS}
    warranty_months = {p[0]: p[4] for p in PRODUCTS}
    for oid, cid, placed, status, carrier, trk, eta, method, note, items in ORDERS:
        total = sum(price[s] * q for s, q in items) + (14.99 if method == "express" else 0.0)
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (oid, cid, placed, status, carrier, trk, eta, method, note, round(total, 2)))
        con.executemany("INSERT INTO order_items VALUES (?,?,?,?)", [(oid, s, q, price[s]) for s, q in items])
    for serial, sku, cid, offset, fw, oid, plan in DEVICES:
        start = TODAY + timedelta(days=offset)
        months = warranty_months[sku] + (EXTENDED_MONTHS if plan == "nimbuscare_plus" else 0)
        coverage = "Manufacturing defects and hardware failures under normal use."
        if plan == "nimbuscare_plus":
            coverage += " NimbusCare+ adds 1 accidental-damage claim and priority replacement."
        con.execute("INSERT INTO devices VALUES (?,?,?,?,?,?)", (serial, sku, cid, start.isoformat(), fw, oid))
        con.execute("INSERT INTO warranties VALUES (?,?,?,?,?,?)",
                    (serial, plan, start.isoformat(), add_months(start, months).isoformat(),
                     CLAIMS.get(serial, 0), coverage))
    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    print(f"Seeded {seed()}")
