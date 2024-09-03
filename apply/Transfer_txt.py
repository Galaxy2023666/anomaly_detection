import pandas as pd

# 读取Excel文件中的特定工作表
excel_file_path = 'D:/新建文件夹/anomaly_detection/result/202406BABP-V4.xlsx'
sheet_name = 'Sheet2'  # 替换为你要导出的工作表名称

# 读取Excel文件中的指定工作表
df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# 导出为TXT文件
txt_file_path = 'D:/新建文件夹/anomaly_detection/result/202406BABP-V4.txt'
df.to_csv(txt_file_path, sep='\t', index=False, header=True, encoding='utf-8')

print(f"工作表 '{sheet_name}' 已成功导出为TXT文件：{txt_file_path}")
