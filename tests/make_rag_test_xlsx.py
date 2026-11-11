#!/usr/bin/env python3
"""Build the folder of workbooks used by tests/run_rag_checks.sh.

  marks.xlsx    not a course sheet at all: student marks, header on row 1,
                to show the pipeline is general
  edge.xlsx     the chunking rules, one tab per case:
                  Batches     title rows, header on row 3, long labels to
                              shorten, one course split over 3 batch rows
                              (merged into one chunk), "-" / en-dash values,
                              a value under a column with no header, a
                              footnote, two identical rows (score ties)
                  Footnotes   the same footnote again (stored once)
                  NumbersTop  numeric rows above the real header
                  List        one column only: no header, every row a note
                  Secret      hidden tab (skipped unless --include-hidden)
                  Empty       nothing at all
                  Unicode     Malayalam and accented text
  UPPER.XLSX    upper-case extension, still found by the folder scan
  books.csv     comma CSV with a UTF-8 BOM, quoted cells holding commas,
                doubled quotes and a line break, a blank line, CRLF ends,
                a "-" cell and a footnote line
  stock.CSV     semicolon CSV (the delimiter is detected), no final newline
  ~$edge.xlsx   an Excel lock file: must be ignored
  readme.txt    not a workbook: must be ignored

Usage: make_rag_test_xlsx.py <output folder>
"""
import os
import sys

from openpyxl import Workbook


def marks(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Semester 1"
    ws.append(["Roll No", "Name", "Department", "Maths", "Physics", "Grade"])
    rows = [
        ("21CS001", "Asha Menon", "CSE", 82, 75, "A"),
        ("21CS002", "Rahul Nair", "CSE", 64, 70, "B"),
        ("21DS003", "Balu Krishna", "Data Science", 91, 88, "S"),
        ("21DS004", "Divya Pillai", "Data Science", 77, 81, "A"),
        ("21EE005", "Farhan Ali", "Electrical", 55, 61, "C"),
        ("21ME006", "Gopi Das", "Mechanical", 68, 72, "B"),
    ]
    for r in rows:
        ws.append(list(r))
    wb.save(path)


def edge(path):
    wb = Workbook()
    b = wb.active
    b.title = "Batches"
    b["A1"] = "Example Institute"
    b["A2"] = "Lab Courses, Autumn"
    b.append(["Code", "Course Name", "Availability (Degree:Prog) Eg. (BTech:DS)",
              "Open as OE? Yes/No", "Instructor email ID, if any", "Slot", "Room"])
    b.append(["LB101", "Physics Lab", "BTech", "No", "Ravi (ravi@x.in)", "PA1", "Lab-1 (B1-B4)"])
    b.append(["LB101", "Physics Lab", "BTech", "No", "Ravi (ravi@x.in)", "PA2", "Lab-1 (B5-B8)"])
    b.append(["LB101", "Physics Lab", "BTech", "No", "Meera (meera@x.in)", "PA3", "Lab-2 (B9-B12)"])
    b.append(["LB205", "Circuits Lab", "-", "–", "Sunil (sunil@x.in)", "PM1", "C-105"])
    b.append(["LB300", "Robotics Workshop", "MTech", "Yes", "Anu (anu@x.in)", "Q", "R-2",
              None, "extra value in a column without a header"])
    b.append(["LB410", "Design Studio", "BTech", "Yes", "Kiran (kiran@x.in)", "E", "D-1"])
    b.append(["LB500", "Thermal Lab", "BTech", "No", "Uma (uma@x.in)", "PA4", "T-9"])
    b.append(["LB410", "Design Studio", "BTech", "Yes", "Kiran (kiran@x.in)", "E", "D-1"])
    b.append([])
    b.append(["The primary instructor approves registration."])

    f = wb.create_sheet("Footnotes")
    f.append(["Code", "Course Name", "Slot"])
    f.append(["FN100", "Seminar", "R1"])
    f.append(["The primary instructor approves registration."])

    n = wb.create_sheet("NumbersTop")
    n.append([2026, 1, 2])
    n.append([10, 20, 30])
    n.append(["Item", "Count", "Price"])
    n.append(["Pens", 40, 12.5])
    n.append(["Books", 7, 350])

    lst = wb.create_sheet("List")
    for item in ["Library opens at 8 am", "Canteen closes at 9 pm", "-"]:
        lst.append([item])

    s = wb.create_sheet("Secret")
    s.append(["Code", "Course Name"])
    s.append(["SC999", "Hidden Course"])
    s.sheet_state = "hidden"

    wb.create_sheet("Empty")

    u = wb.create_sheet("Unicode")
    u.append(["Code", "Course Name", "Instructor"])
    u.append(["ML201", "മലയാളം സാഹിത്യം",
              "José Émile"])
    u.append(["ML202", "Café Management", "Renée"])
    wb.save(path)


def upper(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Main"
    ws.append(["ID", "Item", "Owner"])
    ws.append(["U-7", "Projector", "Admin Office"])
    ws.append(["U-8", "Mechanics Kit", "Physics Store"])
    wb.save(path)


def csv_files(out):
    with open(os.path.join(out, "books.csv"), "wb") as fh:
        fh.write(b"\xef\xbb\xbf"
                 b"Book ID,Title,Author,Shelf,Copies\r\n"
                 b"BK-101,\"Signals, Systems and Noise\",Meera Nair (meera@lib.example),R2-14,3\r\n"
                 b"BK-102,\"The \"\"C\"\" Programming Language\",\"Kernighan,\nRitchie\",R1-02,5\r\n"
                 b"\r\n"
                 b"BK-205,  Linear   Algebra Done Right ,Sheldon Axler,-,1\r\n"
                 b"BK-310,Operating System Concepts,Silberschatz,R3-07,2\r\n"
                 b"Books may be borrowed for 14 days.\r\n")
    with open(os.path.join(out, "stock.CSV"), "wb") as fh:
        fh.write(b"Item;Unit;Price\n"
                 b"Chalk, white;box;45,50\n"
                 b"Duster;piece;30")


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    csv_files(out)
    marks(os.path.join(out, "marks.xlsx"))
    edge(os.path.join(out, "edge.xlsx"))
    upper(os.path.join(out, "UPPER.XLSX"))
    with open(os.path.join(out, "~$edge.xlsx"), "wb") as fh:
        fh.write(b"lock file")
    with open(os.path.join(out, "readme.txt"), "w") as fh:
        fh.write("not a workbook\n")
    print("wrote 3 workbooks, 2 CSV files + 2 files to ignore in", out)


if __name__ == "__main__":
    main()
