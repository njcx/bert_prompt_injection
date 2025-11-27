import json
import csv
import os


def json_to_csv(type_, tag,  json_file_path, csv_file_path):
    try:
        # 读取JSON文件
        print(f"正在读取JSON文件: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        reverse_exposure_data = data.get(type_, [])

        # 定义CSV的列名
        fieldnames = ['prompt', 'tag']

        with open(csv_file_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # 写入表头
            writer.writeheader()
            # 写入数据行
            for item in reverse_exposure_data:
                writer.writerow({
                    'prompt': item.get('prompt', ''),
                    'tag': tag
                })

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{json_file_path}'")
    except json.JSONDecodeError:
        print(f"错误: '{json_file_path}' 不是有效的JSON文件")
    except Exception as e:
        print(f"错误: {str(e)}")


def merge_csv_files(file1, file2, output_file):
    with open(file1, 'r', newline='', encoding='utf-8') as f1, \
         open(file2, 'r', newline='', encoding='utf-8') as f2, \
         open(output_file, 'w', newline='', encoding='utf-8') as out:

        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(out)

        # 读取并写入第一个文件的所有行（包括表头）
        header = next(reader1)  # 获取表头
        writer.writerow(header)
        writer.writerows(reader1)  # 写入 file1 的数据行

        # 跳过第二个文件的表头（假设和第一个一样），只写入数据行
        next(reader2, None)  # 跳过 file2 的表头
        writer.writerows(reader2)


if __name__ == "__main__":
    # 处理包含提示词注入的JSON文件
    input_file = "instruction_attack_scenarios.json"
    path = os.path.join("safety-prompts", input_file)
    json_to_csv('Goal_Hijacking', 1, path, "output1.csv")
    json_to_csv('Prompt_Leaking', 1, path, "output2.csv")
    json_to_csv('Reverse_Exposure', 1, path, "output3.csv")
    json_to_csv('Role_Play_Instruction', 1, path, "output4.csv")

    # 合并CSV文件为一个数据集
    merge_csv_files('output1.csv', 'output2.csv', 'output5.csv')
    merge_csv_files('output3.csv', 'output4.csv', 'output6.csv')
    merge_csv_files('output5.csv', 'output6.csv', 'output7.csv')
    merge_csv_files('output7.csv', 'safe_prompts_20k.csv', 'data_set.csv')
    os.system("rm -rf output*.csv")

    print(f"CSV文件已保存到: data_set.csv")
