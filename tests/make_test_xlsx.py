#!/usr/bin/env python3
"""Build the .xlsx files used by tests/run_xlsx_checks.sh.

  ds_like.xlsx    header on row 5, title rows above, 3 tabs, multi-line cells,
                  blank cells, trailing spaces, a vertically merged cell
  cs_like.xlsx    header on row 1, instructor cells with line breaks
  types.xlsx      numbers, booleans, dates, times, formulas, a hidden tab,
                  an empty tab, special characters, far-right columns
  raw_features.xlsx  hand-written XML for things openpyxl never writes:
                  inline strings, rich text, phonetic runs, _x000D_ escapes,
                  cells and rows without r="" attributes, error cells,
                  the 1904 date system, and a "stored" (uncompressed) entry

Usage: make_test_xlsx.py <output folder>
"""
import datetime
import os
import sys
import zipfile

from openpyxl import Workbook
from openpyxl.styles import Alignment


def ds_like(path):
    wb = Workbook()
    core = wb.active
    core.title = "Core"
    core["A1"] = "Indian Institute of Technology Palakkad"
    core.merge_cells("A1:L1")
    core["A2"] = "Course offerings, Aug-Dec 2026"
    core["A3"] = "Instructions: fill in every column"
    header = ["Code", "Course Name", "Credits", "Open as OE for UG\nYes/No",
              "Open as OE for PG?\nYes/No", "Availability (Deg:Prog:Sem:UpperCap)",
              "Prerequisites? Yes/No. If Yes, mention course name",
              "Primay instructor email ID", "Additional instructor email ID, if any",
              "Slot", "Room", "Grading Scheme"]
    for i, h in enumerate(header, start=1):
        core.cell(row=5, column=i, value=h).alignment = Alignment(wrap_text=True)
    rows = [
        ["DS1010", "Introduction to Data Science and Engineering", "1-0-0-1", "No", "No",
         None, "No", "Abhinandan S P\nabhinandansp@iitpkd.ac.in", None, "R1", "A01-212", "Y-N"],
        ["CS2020A", "Discrete Mathematics", "3-1-0-4", None, None, None, None, None,
         None, "F+I [Fri]", "A01-002 ", None],
        ["DS2010", "Introduction to Optimization", "3-0-0-3", "Yes", "No", "BTech:ALL:SX:XX",
         "MA1011 Linear Algebra and\nOne-variable Calculus\nMA1021 Multivariate calculus (OR) equivalent courses",
         "Sahasranand K R\nsahasranand@iitpkd.ac.in", "-", "C", "A01-002", "Letter Grade"],
        ["DS2030", "Data Structures and Algorithms for Data Science", "3-0-3-5", "Yes", "No",
         "BTech:ALL:SX:20", "ID1110 Introduction to Programming",
         "Sahely Bhadra\nsahely@iitpkd.ac.in", None, "B + PA1", "A01-212 +\nNila CS Lab",
         "Letter Grade"],
        ["DS3010", "Introduction to Machine Learning", "3-0-3-5", "Yes", "No",
         "BTech:ALL:SX:20", "DS2010 & MA2041 <Probability>",
         "Swapnil Hingmire\nswapnilh@iitpkd.ac.in", None,
         "D [Mon,Wed], E [Fri] + PA5", "A01-212 + A01-109", "Letter Grade"],
    ]
    for r, row in enumerate(rows, start=6):
        for c, v in enumerate(row, start=1):
            if v is not None:
                core.cell(row=r, column=c, value=v)
    # Vertically merged "branch" label in column N over three data rows.
    core["N6"] = "DS branch"
    core.merge_cells("N6:N8")

    elec = wb.create_sheet("Elective")
    elec.append(["Code", "Course Name", "Credits", "Slot"])
    elec.append(["DS4010", "Café \u2013 Deep Learning", "3-0-0-3", "A+Q"])
    elec.append(["DS4020", "\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02 NLP", "3-0-0-3", "B"])
    elec.append([])
    elec.append(["DS4030", "Graph  Mining", "3-0-0-3", "   C   "])

    proj = wb.create_sheet("Project Credits")
    proj["A1"] = "Code"
    proj["B1"] = "Credits"
    proj["A2"] = "DS6010"
    proj["B2"] = 6
    wb.save(path)


def cs_like(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Core"
    ws.append(["Code", "Course Name", "Credits", "Primary Instructor Email ID",
               "Additional Instructor Email ID, if any", "Slot", "Room"])
    ws.append(["CS2020A", "Discrete Mathematics", "3-1-0-4",
               "Jayakrishnan M\n(jkmadathil@iitpkd.ac.in)", None, "F+I [Fri]", "A01-002 "])
    ws.append(["CS2111", "Foundations of Computing Systems Lab", "0-0-3-2",
               "Sreejith A V (sreejithav@iitpkd.ac.in)",
               "\nSrimanta Bhattacharya (srimanta@iitpkd.ac.in)", "PA4", "A01-109"])
    ws.append(["CS2013", "Systems Programming", "1-0-3-3",
               "Anish Hirwe (anish@iitpkd.ac.in)", None, "PA2+R2", "A01-109 + A01-102"])
    wb.save(path)


def types(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Numbers & Dates"
    ws.append(["int", 42])
    ws.append(["negative", -17])
    ws.append(["float", 3.14159])
    ws.append(["sum", 0.1 + 0.2])
    ws.append(["big", 123456789012])
    ws.append(["tiny", 1.5e-7])
    ws.append(["bool true", True])
    ws.append(["bool false", False])
    ws.append(["text number", "007"])
    ws.append(["date", datetime.date(2026, 11, 10)])
    ws.append(["datetime", datetime.datetime(2026, 8, 3, 14, 30)])
    ws.append(["datetime sec", datetime.datetime(2026, 8, 3, 14, 30, 15)])
    ws.append(["time", datetime.time(9, 5)])
    ws.append(["old date", datetime.date(1999, 12, 31)])
    ws.append(["leap day", datetime.date(2024, 2, 29)])
    ws.append(["formula", "=A1&B1"])
    ws["A18"] = "far right"
    ws["AB18"] = "col AB"
    ws["ZZ18"] = "col ZZ"
    ws["A20"] = "Tom & Jerry <3 \"quotes\" 'apos'"
    ws["A21"] = "tab\there\u00a0nbsp\u2009thin"

    hidden = wb.create_sheet("Hidden Notes")
    hidden["A1"] = "secret"
    hidden.sheet_state = "hidden"
    wb.create_sheet("Empty")
    wb.save(path)


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<workbookPr date1904="1"/>
<sheets><sheet name="Raw &amp; Odd" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""

WB_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

SHARED = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="4" uniqueCount="4">
<si><t>plain</t></si>
<si><r><t xml:space="preserve">Rich </t></r><r><rPr><b/></rPr><t>text</t></r><r><t xml:space="preserve"> runs</t></r></si>
<si><t>line1_x000D_&#10;line2 &amp; more</t><rPh sb="0" eb="1"><t>PHONETIC-SHOULD-BE-IGNORED</t></rPh></si>
<si><t/></si>
</sst>"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="2"><numFmt numFmtId="164" formatCode="dd\\-mmm\\-yyyy"/><numFmt numFmtId="165" formatCode="0.00&quot; days&quot;"/></numFmts>
<cellStyleXfs count="1"><xf numFmtId="14"/></cellStyleXfs>
<cellXfs count="4"><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="164"/><xf numFmtId="165"/></cellXfs>
</styleSheet>"""

SHEET = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c></row>
<row r="2"><c r="A2" t="inlineStr"><is><t>inline &amp; direct</t></is></c><c r="B2" t="inlineStr"><is><r><t>in</t></r><r><t>line rich</t></r></is></c><c r="C2" t="str"><v>formula &lt;result&gt;</v></c></row>
<row r="3"><c r="A3" t="e"><v>#DIV/0!</v></c><c r="B3" t="b"><v>1</v></c><c r="C3" t="b"><v>0</v></c><c r="D3" s="1"><v>1</v></c></row>
<row r="4"><c r="A4" s="1"><v>1</v></c><c r="B4" s="2"><v>45000</v></c><c r="C4" s="3"><v>12.5</v></c><c r="D4" s="2"><v>45000.75</v></c></row>
<row><c t="inlineStr"><is><t>no r on row or cell</t></is></c><c t="inlineStr"><is><t>second</t></is></c></row>
<row r="9"><c r="C9" t="s"><v>0</v></c><c r="E9" t="inlineStr"><is><t>after gap</t></is></c><c r="F9" s="1"/></row>
<row r="10"><c r="A10"/></row>
</sheetData>
</worksheet>"""


def raw_features(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES, zipfile.ZIP_DEFLATED)
        z.writestr("_rels/.rels", ROOT_RELS, zipfile.ZIP_STORED)       # stored
        z.writestr("xl/workbook.xml", WORKBOOK, zipfile.ZIP_DEFLATED)
        z.writestr("xl/_rels/workbook.xml.rels", WB_RELS, zipfile.ZIP_STORED)
        z.writestr("xl/sharedStrings.xml", SHARED, zipfile.ZIP_DEFLATED)
        z.writestr("xl/styles.xml", STYLES, zipfile.ZIP_DEFLATED)
        z.writestr("xl/worksheets/sheet1.xml", SHEET, zipfile.ZIP_DEFLATED)


def big(path, rows=20000):
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Big")
    ws.append(["Code", "Name", "Credits", "Instructor"])
    for i in range(rows):
        ws.append([f"XX{i:05d}", f"Course number {i} with a longer title",
                   (i % 4) + 1, f"Person {i % 97} (p{i % 97}@example.org)"])
    wb.save(path)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    ds_like(os.path.join(out, "ds_like.xlsx"))
    cs_like(os.path.join(out, "cs_like.xlsx"))
    types(os.path.join(out, "types.xlsx"))
    raw_features(os.path.join(out, "raw_features.xlsx"))
    big(os.path.join(out, "big.xlsx"))
    print("wrote 5 test workbooks to", out)


if __name__ == "__main__":
    main()
