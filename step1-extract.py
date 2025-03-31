# -*- coding: utf-8 -*-
import os
import re
import json
from pathlib import Path

# 将 root_dir 转换为 Path 对象
#root_dir = Path(r"/home/dannyaw/register/registers_e2e_hss/ScriptsHSS/25.7")
root_dir = Path(r"C:\N-5CG2160L1X-Data\dannyaw\Desktop\multi-sim\FT")
output_file = "dataset.json"
dataset = []

def parse_robot_file(robot_file_path):
    """
    解析 .robot 文件中的关键字部分，将每个关键字名称作为 instruction，
    关键字下面的内容作为 response。
    """
    instructions = []
    responses = []
    current_keyword = None
    current_keyword_content = []

    with open(robot_file_path, "r", encoding="utf-8") as rf:
        lines = rf.readlines()

    in_keywords_section = False  # 标记是否进入 *** Keywords *** 部分

    for line in lines:
        line = line.rstrip()  # 去掉右侧的空白字符
        if not line or line.startswith("#"):  # 跳过空行和注释行
            continue

        # 检测到 *** Keywords *** 行
        if line.startswith("*** Keywords ***"):
            in_keywords_section = True
            continue

        # 如果未进入 *** Keywords *** 部分，跳过
        if not in_keywords_section:
            continue

        # 检测关键字（顶格写的行）
        if re.match(r"^\S+", line):  # 顶格写的非空行
            # 保存上一个关键字的内容
            if current_keyword:
                instructions.append(current_keyword)
                responses.append("\n".join(current_keyword_content).strip())

            # 开始新的关键字
            current_keyword = line.strip()  # 当前行作为关键字名称
            current_keyword_content = []  # 重置内容列表
        elif line.startswith("    ") or line.startswith("\t"):  # 内容行（4个空格或Tab缩进）
            current_keyword_content.append(line.strip())

    # 保存最后一个关键字的内容
    if current_keyword:
        instructions.append(current_keyword)
        responses.append("\n".join(current_keyword_content).strip())

    return instructions, responses

def parse_tsv_file(tsv_file, dataset):
    """
    解析 .tsv 文件，将 [Documentation] 行作为 instruction，
    并解析对应的 .robot 文件内容。
    """
    with open(tsv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_case = None
    instruction = None
    test_steps = []  # 使用列表作为引用类型

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "*Keyword*" in line:
            print(f"Stopping processing for {tsv_file} due to *Keyword*")
            break

        # 使用正则表达式匹配 [Documentation] 行
        match = re.match(r"^(.*?)\s*\[Documentation\]\s*(.*)$", line)
        if match:
            # 保存上一个用例
            if current_case:
                dataset.append({
                    "instruction": instruction,
                    "response": "\n".join(test_steps)
                })

            # 开始新用例
            current_case = match.group(1).strip()  # 用例名称是 [Documentation] 前面的部分
            instruction = match.group(2).strip()  # 提取 [Documentation] 后面的内容
            test_steps = []  # 重置为列表
        else:
            # 累积当前用例的步骤
            test_steps.append(line)

    # 保存最后一个用例
    if current_case:
        dataset.append({
            "instruction": instruction,
            "response": "\n".join(test_steps)
        })


def main():
    # 解析所有 .tsv 文件
    tsv_files = list(root_dir.rglob("*.tsv"))
    if not tsv_files:
        print("No .tsv files found in the directory or its subdirectories.")
        exit(1)

    for tsv_file in tsv_files:
        print(f"Processing TSV file: {tsv_file}")
        parse_tsv_file(tsv_file, dataset)

        # 查找对应的 .robot 文件内容
        robot_dir = tsv_file.parent
        print(f"{robot_dir}")
        robot_files = list(robot_dir.rglob("*.robot"))
        print(f"{robot_files}")
        for robot_file in robot_files:
            instructions, responses = parse_robot_file(robot_file)
            for instr, resp in zip(instructions, responses):
                dataset.append({
                    "instruction": instr,
                    "response": resp
                })

    # 保存数据集到 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Generated dataset with {len(dataset)} samples")

main()