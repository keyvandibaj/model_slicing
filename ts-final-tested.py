import re
import time
import subprocess
import os

# --------------------------
# Configuration
# --------------------------
LOG_FILE = 'anomalies.log'
LINES_TO_CHECK = 50  # Process 50-line batches
ANOMALY_THRESHOLD = 5  # UEID must appear ≥5 times in batch
CHECK_INTERVAL = 2
SINR_THRESHOLD = -2  # Only run gRPC if neighbor avg SINR > -2

# Regex to extract UEID, Serving SINR, and Neighbor info
UEID_REGEX = re.compile(r'UE ID ([\d.]+)')
SERVING_SINR_REGEX = re.compile(r'Serving SINR: (-?\d+\.?\d*)')
NEIGHBOR_REGEX = re.compile(r'Neighbor (\d+): ID=([0-9.]+), SINR=([0-9.-]+)')

# gRPC command template
GRPCURL_PATH = "grpcurl"
COMMAND_TEMPLATE = '''{grpcurl} -plaintext -d "{{ \\
    \\"e2NodeID\\": \\"36000000\\", \\
    \\"plmnID\\": \\"313131\\", \\
    \\"ranName\\": \\"gnb_131_133_31000000\\", \\
    \\"RICE2APHeaderData\\": {{ \\
        \\"RanFuncId\\": 300, \\
        \\"RICRequestorID\\": 1001 \\
    }}, \\
    \\"RICControlHeaderData\\": {{ \\
        \\"ControlStyle\\": 3, \\
        \\"ControlActionId\\": 1, \\
        \\"UEID\\": \\"{ueid}\\" \\
    }}, \\
    \\"RICControlMessageData\\": {{ \\
        \\"RICControlCellTypeVal\\": 4, \\
        \\"TargetCellID\\": \\"{target_cell}\\" \\
    }}, \\
    \\"RICControlAckReqVal\\": 0 \\
}}" 10.244.0.120:7777 rc.MsgComm.SendRICControlReqServiceGrpc'''

# --------------------------
# Step 3: Extract UEID, Serving SINR, and Neighbor Data
# --------------------------
def extract_ueid_neighbor_data(lines):
    ueid_neighbor_data = {}
    ueid_serving_sinrs = {}  # Store serving SINR per UEID
    ueid_count = {}

    for line in lines:
        ue_match = UEID_REGEX.search(line)
        if not ue_match:
            continue

        # Handle decimal UEID like "23.0" → "23"
        ueid_str = ue_match.group(1).strip()
        try:
            ueid = int(float(ueid_str))
        except ValueError:
            continue

        # Count how many times this UEID appears in batch
        ueid_count[ueid] = ueid_count.get(ueid, 0) + 1

        # Extract Serving SINR
        serving_match = SERVING_SINR_REGEX.search(line)
        if serving_match:
            try:
                serving_sinr = float(serving_match.group(1))
                if ueid not in ueid_serving_sinrs:
                    ueid_serving_sinrs[ueid] = []
                ueid_serving_sinrs[ueid].append(serving_sinr)
            except ValueError:
                continue

        # Parse neighbor data
        neighbors = NEIGHBOR_REGEX.findall(line)
        if not neighbors:
            continue

        if ueid not in ueid_neighbor_data:
            ueid_neighbor_data[ueid] = {}

        for _, neighbor_id_str, sinr_str in neighbors:
            try:
                neighbor_id = int(float(neighbor_id_str))
                sinr = float(sinr_str)
            except ValueError:
                continue

            if neighbor_id <= 0:
                continue

            if neighbor_id not in ueid_neighbor_data[ueid]:
                ueid_neighbor_data[ueid][neighbor_id] = []

            ueid_neighbor_data[ueid][neighbor_id].append(sinr)

    return ueid_neighbor_data, ueid_serving_sinrs, ueid_count

# --------------------------
# Step 4: Run gRPC Command with Delay
# --------------------------
def run_grpc_command(ueid, best_neighbor_id, neighbor_avg, serving_avg):
    formatted_ueid = f"{ueid:05d}"
    target_cell = f"1110{best_neighbor_id}"
    print(f"[INFO] Running command for UEID: {formatted_ueid}, TargetCellID: {target_cell}")
    print(f"[INFO] Neighbor Avg: {neighbor_avg:.2f}, Serving Avg: {serving_avg:.2f}")

    try:
        command = COMMAND_TEMPLATE.format(
            grpcurl=GRPCURL_PATH,
            ueid=formatted_ueid,
            target_cell=target_cell
        )
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] gRPC failed for UEID {formatted_ueid}: {e}")
    finally:
        print("[INFO] Waiting 5 seconds before next command...")
        time.sleep(5)
# --------------------------
# Step 5: Select Best Neighbor by Average SINR
# --------------------------
def get_best_neighbor_id(avg_sinrs):
    if not avg_sinrs:
        return None

    # Filter neighbors with avg SINR > -2
    valid_neighbors = {k: v for k, v in avg_sinrs.items() if v > SINR_THRESHOLD}
    if not valid_neighbors:
        return None

    return max(valid_neighbors, key=valid_neighbors.get)
# --------------------------
# Main Execution Loop
# --------------------------
def main():
    print("[INFO] Starting tsf.py...")
    current_position = 0  # Start from beginning of log file
    processed_ueids = set()  # Track UEs already processed

    while True:
        if not os.path.exists(LOG_FILE):
            print(f"[INFO] Log file '{LOG_FILE}' not found.")
            time.sleep(CHECK_INTERVAL)
            continue

        with open(LOG_FILE, 'r') as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        remaining_lines = total_lines - current_position

        # Ensure we have at least 50 lines from current_position
        if remaining_lines < LINES_TO_CHECK:
            print(f"[INFO] Not enough lines ({remaining_lines}) at position {current_position}. Waiting for new data...")
            time.sleep(CHECK_INTERVAL)
            continue

        # Always process 50-line batch
        batch_lines = all_lines[current_position:current_position + LINES_TO_CHECK]
        print(f"[DEBUG] Parsing {len(batch_lines)} lines from position {current_position}")

        ueid_neighbor_data, ueid_serving_sinrs, ueid_count = extract_ueid_neighbor_data(batch_lines)

        for ueid in ueid_count:
            if ueid_count.get(ueid, 0) < ANOMALY_THRESHOLD:
                continue

            # Compute average neighbor SINR
            if ueid not in ueid_neighbor_data:
                print(f"[INFO] Skipping UEID {ueid}: No neighbor data")
                continue

            avg_neighbor_sinrs = {
                n_id: sum(sinr_list) / len(sinr_list)
                for n_id, sinr_list in ueid_neighbor_data[ueid].items()
            }

            best_neighbor_id = get_best_neighbor_id(avg_neighbor_sinrs)
            if not best_neighbor_id:
                continue

            best_neighbor_avg = avg_neighbor_sinrs[best_neighbor_id]

            # Compute average serving SINR
            if ueid not in ueid_serving_sinrs:
                print(f"[INFO] Skipping UEID {ueid}: No serving SINR data")
                continue

            serving_list = ueid_serving_sinrs[ueid]
            if not serving_list:
                continue

            avg_serving_sinr = sum(serving_list) / len(serving_list)

            print(f"[INFO] Best neighbor for UEID {ueid}: {best_neighbor_id} (avg SINR: {best_neighbor_avg:.2f}), Serving avg: {avg_serving_sinr:.2f}")

            # ✅ Only run if neighbor signal is better than serving
            if best_neighbor_avg > avg_serving_sinr:
                if ueid in processed_ueids:
                    print(f"[INFO] Skipping UEID {ueid}: Already processed.")
                    continue

                run_grpc_command(ueid, best_neighbor_id, best_neighbor_avg, avg_serving_sinr)
                processed_ueids.add(ueid)
            else:
                print(f"[INFO] Skipping UEID {ueid}: Neighbor SINR ({best_neighbor_avg:.2f}) ≤ Serving SINR ({avg_serving_sinr:.2f})")

        # Move to next 50-line batch
        current_position += LINES_TO_CHECK
        print(f"[INFO] Batch complete. Next batch starts at line {current_position}")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
