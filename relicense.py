# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

import os

# 1. 새로운 Apache 2.0 + 특허 고지 헤더 정의
NEW_HEADER = """# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

"""

IGNORE_DIRS = {'.venv', 'venv', '.git', '__pycache__', 'build', 'dist'}

def update_file_header(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. 기존 헤더 건너뛰기 로직
    # '#'으로 시작하는 주석 라인이나 빈 줄이 끝날 때까지 스캔
    start_index = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Shebang은 보존하고 싶다면 아래 주석 해제
        # if i == 0 and stripped.startswith('#!'): continue
        
        if stripped.startswith('#') or not stripped:
            start_index = i + 1
        else:
            # 주석이 아닌 실제 코드나 Docstring을 만나면 중단
            break
            
    # 3. 새로운 헤더 + 기존 코드 결합
    content_after_header = lines[start_index:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(NEW_HEADER)
        f.writelines(content_after_header)
    print(f"✅ Relicensed: {file_path}")

def main():
    for root, _, files in os.walk('.'):
        if any(ignore in root.split(os.sep) for ignore in IGNORE_DIRS):
                continue
        for file in files:
            if file.endswith('.py') and file != 'relicense_versor.py':
                update_file_header(os.path.join(root, file))

if __name__ == "__main__":
    main()