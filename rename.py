import os

PATH = "./"  # 定义文件所在目录
jpg_ETX = ".png"  # 定义文件后缀
file_id = 0
'''重命名处理函数'''


def rename_file(oldname):
    global file_id
    newname = file_id
    oldname = PATH + oldname  # PATH是路径,os.sep为当前系统的目录分割符,不同系统不一样,例如windows就是\,本系统是linux所以是/,oldname是文件名
    newname ="./" + newname + jpg_ETX  # 加上jpg后缀的新文件名
    os.rename(oldname, newname)  # 调用os.rename重命名函数
    print("oldname  -------------> newname  ", oldname, newname)
    file_id += 1


if __name__ == "__main__":

    fileList = os.listdir(PATH)  # 读取PATH目录中所有文件
    fileList.sort()  # 对文件名进行排序
    for name in fileList:
        if ".png" in name:
            rename_file(name)  # 重命名
