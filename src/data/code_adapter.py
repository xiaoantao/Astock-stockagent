import re
SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")
def normalize(code : str):
    prefixes = ""
    if code in ("","  ",None):
        raise ValueError(f"股票代码不能为空:{code!r}")
    if code.lower().startswith("jp.") or code.lower().endswith(".jp"):
        raise ValueError(f"无法识别的代码编号:{code!r}")
    pattern = r"\d+"
    code_normal = re.search(pattern,code)
    if code_normal:
        prefixes = code_normal.group()
    if len(prefixes) != 6:
        raise ValueError(f"非6位数字无法进行识别:{code!r}")
    prefixes3 = prefixes[:3]
    if prefixes3 in SH_PREFIXES:
        return prefixes+".SH"
    elif prefixes3 in SZ_PREFIXES:
        return prefixes+".SZ"
    else:
        raise ValueError(f"六位股票代码不在范围内:{code!r}")


def to_baostock(code : str):
    prefixes = ""
    if code in ("","  ",None):
        raise ValueError(f"股票代码不能为空:{code!r}")
    if code.lower().startswith("jp.") or code.lower().endswith(".jp"):
        raise ValueError(f"无法识别的代码编号:{code!r}")
    pattern = r"\d+"
    code_normal = re.search(pattern,code)
    if code_normal:
        prefixes = code_normal.group()
    if len(prefixes) != 6:
         raise ValueError(f"非6位数字无法进行识别:{code!r}")
    prefixes3 = prefixes[:3]
    if prefixes3 in SH_PREFIXES:
         return "sh."+prefixes
    elif prefixes3 in SZ_PREFIXES:
         return "sz."+prefixes
    else:
         raise ValueError(f"六位股票代码不在范围内:{code!r}")



def to_akshare(code : str):
    prefixes = ""
    if code in ("","  ",None):
        raise ValueError(f"股票代码不能为空:{code!r}")
    if code.lower().startswith("jp.") or code.lower().endswith(".jp"):
        raise ValueError(f"无法识别的代码编号:{code!r}")
    pattern = r"\d+"
    code_normal = re.search(pattern,code)
    if code_normal:
        prefixes = code_normal.group()
    if len(prefixes) != 6:
        raise ValueError(f"非6位数字无法进行识别:{code!r}")
    prefixes3 = prefixes[:3]
    if prefixes3 in SH_PREFIXES:
        return prefixes
    elif prefixes3 in SZ_PREFIXES:
        return prefixes
    else:
        raise ValueError(f"六位股票代码不在范围内:{code!r}")

def to_tushare(code : str):
    prefixes = ""
    if code in ("","  ",None):
        raise ValueError(f"股票代码不能为空:{code!r}")
    if code.lower().startswith("jp.") or code.lower().endswith(".jp"):
        raise ValueError(f"无法识别的代码编号:{code!r}")
    pattern = r"\d+"
    code_normal = re.search(pattern,code)
    if code_normal:
        prefixes = code_normal.group()
    if len(prefixes) != 6:
        raise ValueError(f"非6位数字无法进行识别:{code!r}")
    prefixes3 = prefixes[:3]
    if prefixes3 in SH_PREFIXES:
        return prefixes+".SH"
    elif prefixes3 in SZ_PREFIXES:
        return prefixes+".SZ"
    else:
        raise ValueError(f"六位股票代码不在范围内:{code!r}")

