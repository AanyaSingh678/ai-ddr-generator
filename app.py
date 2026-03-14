from parser.pdf_parser import extract_text

inspection_path = r"C:\Users\Aanya Singh\Projects\ai-ddr-generator\data\Inspection Report.pdf"

print("Reading inspection report...")

inspection_text = extract_text(inspection_path)

print("\n--- Extracted Text Preview ---\n")

print(inspection_text[:1000])
