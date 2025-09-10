import pandas as pd
import re

# Load dataset
df = pd.read_csv("run1_3rtu_2s.csv")

# --- Parsers ---
def parse_tcp(info):
    pat = re.compile(
        r"(\d+)\s*>\s*(\d+)\s*\[(.*?)\]\s*Seq=(\d+)(?:\s*Ack=(\d+))?\s*Win=(\d+)\s*Len=(\d+)(?:\s*MSS=(\d+))?"
    )
    m = pat.match(info.strip())
    if m:
        return {
            "SrcPort": m.group(1),
            "DstPort": m.group(2),
            "Flags": m.group(3),
            "Seq": m.group(4),
            "Ack": m.group(5),
            "Win": m.group(6),
            "Len": m.group(7),
            "MSS": m.group(8),
        }
    return {}

def parse_modbus(info):
    pat = re.compile(r"(.*?)Trans:\s*(\d+);\s*Unit:\s*(\d+),\s*Func:\s*(\d+):\s*(.*)")
    m = pat.match(info.strip())
    if m:
        return {
            "Direction": m.group(1).strip(),
            "TransID": m.group(2),
            "UnitID": m.group(3),
            "FuncCode": m.group(4),
            "FuncDesc": m.group(5),
        }
    return {}

def parse_generic(info):
    return {"Other Protocol Details": info}

# Dispatcher
def parse_row(row):
    proto = row["Protocol"]
    info = str(row["Info"])
    if proto == "TCP":
        return parse_tcp(info)
    elif proto == "Modbus/TCP":
        return parse_modbus(info)
    else:
        return parse_generic(info)

# Apply parsers
parsed = df.apply(parse_row, axis=1, result_type="expand")

# Merge into unified dataset
final_df = pd.concat([df, parsed], axis=1)

# Save final dataset
final_df.to_csv("processed_dataset5.csv", index=False)
