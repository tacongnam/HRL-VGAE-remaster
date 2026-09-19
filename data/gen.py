import copy
import json
from pathlib import Path
import random
import sys


def generate_permutations(input_path_str: str, num_samples: int):
    input_path = Path(input_path_str).resolve()
    if not input_path.is_file():
        print(f"Lỗi: Không tìm thấy file tại '{input_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Đang đọc dữ liệu từ: {input_path.name}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_requests = data.get("R", [])
    total_reqs = len(original_requests)
    print(f"Tổng số request: {total_reqs}")

    if total_reqs == 0:
        print("Lỗi: Danh sách 'R' trống!", file=sys.stderr)
        sys.exit(1)

    timestamps = [req["T"] for req in original_requests]
    parent_dir = input_path.parent
    base_stem = input_path.stem

    for i in range(1, num_samples + 1):
        shuffled_requests = copy.deepcopy(original_requests)
        random.shuffle(shuffled_requests)
        for t, req in zip(timestamps, shuffled_requests):
            req["T"] = t
        new_data = {
            "V": data.get("V"),
            "E": data.get("E"),
            "F": data.get("F"),
            "R": shuffled_requests,
        }
        output_path = parent_dir / f"{base_stem}_permuted_{i}.json"
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(new_data, out_f, indent=2, ensure_ascii=False)

        print(f"[{i}/{num_samples}] Đã sinh file: {output_path.name}")

    print("\nHoàn tất sinh toàn bộ samples!")


# ĐÃ SỬA: Thay _name_ và _main_ bằng __name__ và __main__
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: uv run gen.py <đường_dẫn_file_gốc> <số_sample>")
        sys.exit(1)

    file_url = sys.argv[1]
    try:
        samples = int(sys.argv[2])
        if samples <= 0:
            raise ValueError
    except ValueError:
        print(
            "Lỗi: Số lượng sample phải là số nguyên dương lớn hơn 0!",
            file=sys.stderr,
        )
        sys.exit(1)

    generate_permutations(file_url, samples)