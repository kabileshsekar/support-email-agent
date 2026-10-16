"""Read-only access to the Device & Order database."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional

from .config import DB_PATH


class SupportDB:
    def __init__(self, path: Path = DB_PATH):
        if not path.exists():
            from data.seed import seed
            seed(path)
        self.path = path

    def _q(self, sql: str, params: tuple = ()) -> list[dict]:
        with closing(sqlite3.connect(self.path)) as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        rows = self._q(sql, params)
        return rows[0] if rows else None

    # customers
    def customer_by_email(self, email: str) -> Optional[dict]:
        return self._one("SELECT * FROM customers WHERE lower(email) = lower(?)", (email.strip(),))

    def all_customer_emails(self) -> set[str]:
        return {r["email"].lower() for r in self._q("SELECT email FROM customers")}

    # orders
    def order(self, order_id: str) -> Optional[dict]:
        o = self._one("SELECT * FROM orders WHERE upper(order_id) = upper(?)", (order_id.strip(),))
        if o:
            o["items"] = self._q(
                "SELECT p.name AS product, p.sku, i.quantity, i.unit_price FROM order_items i "
                "JOIN products p ON p.sku = i.sku WHERE i.order_id = ?", (o["order_id"],))
        return o

    def orders_for_customer(self, customer_id: str) -> list[dict]:
        rows = self._q("SELECT order_id FROM orders WHERE customer_id = ? ORDER BY placed_date DESC", (customer_id,))
        return [self.order(r["order_id"]) for r in rows]

    # devices & warranty
    def device(self, serial: str) -> Optional[dict]:
        return self._one(
            "SELECT d.*, p.name AS product, p.category, p.latest_firmware FROM devices d "
            "JOIN products p ON p.sku = d.sku WHERE upper(d.serial_number) = upper(?)", (serial.strip(),))

    def devices_for_customer(self, customer_id: str) -> list[dict]:
        return self._q(
            "SELECT d.*, p.name AS product, p.category, p.latest_firmware FROM devices d "
            "JOIN products p ON p.sku = d.sku WHERE d.customer_id = ? ORDER BY d.purchase_date DESC", (customer_id,))

    def warranty(self, serial: str) -> Optional[dict]:
        return self._one("SELECT * FROM warranties WHERE upper(serial_number) = upper(?)", (serial.strip(),))

    # products
    def products(self) -> list[dict]:
        rows = self._q("SELECT * FROM products")
        for r in rows:
            r["specs"] = json.loads(r["specs"])
        return rows

    def find_product(self, query: str) -> Optional[dict]:
        """Match by SKU or by the best word overlap with the product name/category."""
        q = query.lower()
        words = set(re.findall(r"[a-z0-9]+", q)) - {"nimbus", "the", "a", "my"}
        best, best_score = None, 0
        for p in self.products():
            if p["sku"].lower() in q:
                return p
            name_words = set(re.findall(r"[a-z0-9]+", f"{p['name']} {p['category']}".lower())) - {"nimbus"}
            score = len(words & name_words)
            if score > best_score:
                best, best_score = p, score
        return best
