#!/usr/bin/python
# -*- coding: utf-8 -*-
import re

##########################
# 发版改这里的版本号, 会自动打tag & 自动编译py3.7～3.10的 wheel package 到 指定pypi服务器
# 默认 MR 和 Master 不发版情况下，只做py310的编译过程, 每个 MR 都跑 py3.7～3.10 的编译过程的话，速度会比较慢
# zhilight release version
__version__ = "0.4.16"

##########################

VersionPattern = "^[v]?[0-9]{1,}[.][0-9]{1,}[.][0-9]{1,}$"


def is_newer_version(tags):
    vers = tags.split(" ")
    for v in vers:
        if _parse_version(v) is None:
            continue
        if not _cmp_version(__version__, v):
            return False
    return True


def _cmp_version(v1, v2):
    a = _parse_version(v1)
    b = _parse_version(v2)
    if a is None or b is None:  # skip
        return False
    for i in range(len(a)):
        if a[i] == b[i]:
            continue
        elif a[i] > b[i]:
            return True
        else:
            return False
    return False


def _parse_version(version):
    ver = version.strip()
    if re.match(VersionPattern, ver) is None:
        return None
    ver = ver[1:] if ver.startswith("v") else ver
    tokens = ver.split(".")
    return int(tokens[0]), int(tokens[1]), int(tokens[2])


if __name__ == "__main__":
    tags = "v3.1.0 v2.9.9 v3.9.7, v4.9.1.dev0"
    print(is_newer_version(tags))
