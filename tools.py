# -*- coding:utf-8
import os
import stat
import magic
import codecs
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_binary_file_1(file_path):
    '''
    根据text文件数据类型判断是否是二进制文件
    :param file_path: 文件名（含路径）
    :return: True或False，返回是否是二进制文件
    '''
    TEXT_BOMS = (
        codecs.BOM_UTF16_BE,
        codecs.BOM_UTF16_LE,
        codecs.BOM_UTF32_BE,
        codecs.BOM_UTF32_LE,
        codecs.BOM_UTF8,
    )
    with open(file_path, 'rb') as file:
        CHUNKSIZE = 8192
        initial_bytes = file.read(CHUNKSIZE)
        file.close()
    #: BOMs to indicate that a file is a text file even if it contains zero bytes.
    return not any(initial_bytes.startswith(bom) for bom in TEXT_BOMS) and b'\0' in initial_bytes


def is_binary_file_2(ff):
    '''
    根据magic文件的魔术判断是否是二进制文件
    :param ff: 文件名（含路径）
    :return: True或False，返回是否是二进制文件
    '''
    mime_kw = 'x-executable|x-sharedlib|octet-stream|x-object'  ###可执行文件、链接库、动态流、对象
    try:
        magic_mime = magic.from_file(ff, mime=True)
        magic_hit = re.search(mime_kw, magic_mime, re.I)
        if magic_hit:
            return True
        else:
            return False
    except Exception as e:
        return False


# 判断文件是否是elf文件
def is_ELFfile(filepath):
    if not os.path.exists(filepath):
        # logger.info('file path {} doesnot exits'.format(filepath))
        return False
    # 文件可能被损坏，捕捉异常
    try:
        FileStates = os.stat(filepath)
        FileMode = FileStates[stat.ST_MODE]
        if not stat.S_ISREG(FileMode) or stat.S_ISLNK(FileMode):  # 如果文件既不是普通文件也不是链接文件
            return False
        with open(filepath, 'rb') as f:
            header = (bytearray(f.read(4))[1:4]).decode(encoding="utf-8")
            # logger.info("header is {}".format(header))
            if header in ["ELF"]:
                # print header
                return True
    except UnicodeDecodeError as e:
        # logger.info("is_ELFfile UnicodeDecodeError {}".format(filepath))
        # logger.info(str(e))
        pass

    return False


def is_binary_file(path):
    return any((is_ELFfile(path), is_binary_file_1(path), is_binary_file_2(path)))


def count_line(path):
    sum = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            tmp_file = os.path.join(root, name)
            if tmp_file.find('venv') != -1:
                continue
            if tmp_file.find('idea') != -1:
                continue
            if tmp_file.find('__pycache__') != -1:
                continue
            if tmp_file.find('.git') != -1:
                continue
            tmp = 0
            if is_binary_file(tmp_file):
                continue
            for index, line in enumerate(open(tmp_file, 'r', encoding='utf8')):
                tmp += 1
            # print(tmp_file, ":", tmp)
            sum += tmp
    return sum


if __name__ == '__main__':
    print(count_line("."))
