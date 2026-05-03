# Carwash Main Altar Digital Management System (CMDMS)

> A Python-based digital management system built for the **Ministry of Repentance and Holiness, Carwash Main Altar — Migosi Region, Kisumu, Kenya.**

---

## Overview

CMDMS replaces manual record-keeping with a clean, menu-driven digital system covering four core areas of ministry operations:

| Module | Description |
|---|---|
| 👥 Member Management | Register and view all ministry members |
| 📋 Attendance | Record and summarize service attendance |
| 💰 Finance | Track income, expenses, and running balance in KES |
| 📅 Operations | Manage events, tasks, and assignments with status tracking |

All data is stored locally in lightweight JSON files — no internet connection or database required.

---

## Project Structure

```
carwash-main-altar-dms/
│
├── cmdms.py            # Main application — run this
├── requirements.txt    # Dependencies (Python stdlib only)
├── .gitignore          # Excludes data files from version control
└── data/               # Auto-created on first run
    ├── members.json
    ├── attendance.json
    ├── finance.json
    └── operations.json
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/jameskoero/carwash-main-altar-dms.git
cd carwash-main-altar-dms
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the system
```bash
python cmdms.py
```

The `data/` folder and all JSON files are created automatically on first run.

> **On Android (PyramIDE):** Open `cmdms.py` and tap Run. The menu will appear in the terminal.

---

## Features

### 👥 Member Management
- Add members with name, phone, and role
- View full membership list with join dates
- Auto-generated member IDs

### 📋 Attendance
- Record attendance per service session (Sunday, Midweek, Special)
- Mark each member present or absent
- View all historical records
- Summary: total sessions, total count, average attendance

### 💰 Finance
- Record income by source (Tithe, Offering, Donation, etc.)
- Record expenses by category (Supplies, Welfare, Transport, etc.)
- Full finance report with income, expenses, and KES balance
- Surplus / Deficit indicator

### 📅 Operations & Events
- Add events and tasks with date, category, and assignment
- Track status: Pending → In Progress → Completed
- View all events with full detail

### 📊 Dashboard
- At-a-glance summary of members, sessions, balance, and pending tasks

---

## Data Storage

All records are stored in JSON format inside the `data/` folder:

```json
// Example: finance.json
{
  "income": [
    { "date": "2025-04-01", "source": "Offering", "amount": 5000.0, "note": "Sunday service" }
  ],
  "expenses": [
    { "date": "2025-04-02", "category": "Supplies", "amount": 1200.0, "note": "Communion" }
  ]
}
```

> The `data/` folder is listed in `.gitignore` to protect ministry records from being published publicly.

---

## Requirements

- Python 3.6 or higher
- No third-party libraries required — uses Python standard library only

---

## Screenshots

*Coming soon — terminal screenshots of dashboard and finance report*

---

## Roadmap

- [ ] Export reports to PDF using ReportLab
- [ ] Search members by name or role
- [ ] Monthly finance summary charts (matplotlib)
- [ ] Password-protected admin login
- [ ] SMS notification integration (Africa's Talking API)

---

## Author

**James Koero (Jayalo)**
BSc Physics & Mathematics | Self-Taught ML Engineer | Preacher
📍 Kisumu, Kenya
📧 jmskoero@gmail.com
🔗 [github.com/jameskoero](https://github.com/jameskoero)

---

## About

Built to serve the Ministry of Repentance and Holiness, Carwash Main Altar, Migosi Region — replacing paper records with a reliable, offline-first digital system.

*"Whatever you do, work at it with all your heart." — Colossians 3:23*
