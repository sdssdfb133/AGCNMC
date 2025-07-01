# import torch
# print(torch.__version__)                # 查看pytorch安装的版本号
# print(torch.cuda.is_available())        # 查看cuda是否可用。True为可用，即是gpu版本pytorch
# print(torch.cuda.get_device_name(0))    # 返回GPU型号
# print(torch.cuda.device_count())        # 返回可以用的cuda（GPU）数量，0代表一个
# print(torch.version.cuda)               # 查看cuda的版本
# import torch_geometric

import csv

# # 输入txt文件路径
# input_file = 'dd_delete.txt'
# # md_delete.txt
# # mm_delete.txt
# # dd_delete.txt
# # 输出csv文件路径
# output_file = 'd-d.csv'
# # m-d.csv
# #m-m.csv
# #d-d.csv
# # 打开并读取txt文件
# with open(input_file, 'r') as txt_file:
#     lines = txt_file.readlines()

# # 处理txt文件中的每一行，并写入新生成的csv文件
# with open(output_file, 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
    
#     for line in lines:
#         # 假设每行的内容是以空格或制表符分隔的，可以根据实际情况修改分隔符
#         row = line.strip().split()  # 默认按空格分割
#         writer.writerow(row)
###########################################################################
# def number_lines(input_file, output_file):
#     try:
#         with open(input_file, 'r', encoding='utf-8') as infile, \
#              open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            
#             # 创建一个csv写入对象，指定字段名
#             csv_writer = csv.writer(outfile)
            
#             # 从0开始编号，并写入csv文件
#             for line_number, line in enumerate(infile):
#                 # 去掉行尾的换行符，将编号和文本作为两个字段写入csv
#                 csv_writer.writerow([line_number, line.strip()])
        
#         print(f"Lines numbered and saved to {output_file}")
        
#     except FileNotFoundError:
#         print(f"Error: {input_file} not found.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # disname.txt
# # mirname.txt
# # 输入文件名和输出文件名
# input_filename = 'mirname.txt'
# output_filename = 'mir_name.csv'

# # 调用函数进行行编号
# number_lines(input_filename, output_filename)
#################################################################
# import csv

# # 输入和输出文件名
# input_file = 'm-d.csv'
# output_file = 'result_data.csv'

# # 打开并读取输入CSV文件
# with open(input_file, mode='r') as infile:
#     reader = csv.reader(infile)
#     data = list(reader)

# # 打开并写入输出CSV文件
# with open(output_file, mode='w', newline='') as outfile:
#     writer = csv.writer(outfile)
    
#     # 遍历数据，找到值为1的行和列
#     for i, row in enumerate(data):
#         for j, value in enumerate(row):
#             if value == '1':  # 确保这里是字符串'1'
#                 writer.writerow([i, j])

# print(f"处理完成，结果已保存至 {output_file}")
import csv
import random
import os

# 输入文件名和输出文件名
input_file = 'result_data.csv'
output_file_1 = 'result_test.csv'
output_file_2 = 'result_one.csv'

# 读取原始数据
with open(input_file, mode='r') as infile:
    reader = csv.reader(infile)
    data = list(reader)

# 打乱数据
random.shuffle(data)

# 按照1:4的比例分割数据
split_index = len(data) // 5  # 1比4分割，计算出1部分的大小

# 分割数据
data_1 = data[:split_index]
data_2 = data[split_index:]

# 写入到新的CSV文件
with open(output_file_1, mode='w', newline='') as outfile1:
    writer = csv.writer(outfile1)
    writer.writerows(data_1)

with open(output_file_2, mode='w', newline='') as outfile2:
    writer = csv.writer(outfile2)
    writer.writerows(data_2)

print(f"数据已经打乱并分割，结果已保存至 {output_file_1} 和 {output_file_2}")
