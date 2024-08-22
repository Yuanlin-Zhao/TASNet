import os

# 指定图片所在文件夹的路径
folder_path = r'E:\zyl\LGTD-main\LGTD-main\Jilin-189\trainset\training_set\GT\003'

# 获取文件夹下所有图片文件的名称并按原顺序保存到列表中
file_list = []
for i in range(1550):
    file_list.append(i)
# 将图片文件名保存到txt文件中
with open(r'E:\zyl\LGTD-main\LGTD-main\Jilin-189\trainset\A.txt', 'w') as f:
    for file_name in file_list:
        if file_name < 10:
            name = '0000000' + str(file_name)
        if 10 <= file_name < 100:
            name = '000000' + str(file_name)
        if 100 <= file_name < 1000:
            name = '00000' + str(file_name)
        if 1000 <= file_name < 10000:
            name = '0000' + str(file_name)
        if 10000 <= file_name < 100000:
            name = '000' + str(file_name)
        filemana= '003/' + name + '.png'
        f.write(filemana + '\n')

print('图片文件名已按原顺序保存到image_names.txt文件中')