# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:55
# @File : common.py
# @Last Modify Time : 2018/1/30 15:55
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  common.py
    FUNCTION : Common File
"""

seed_num = 666
unkkey = "<unk>"
paddingkey = "<pad>"

death = 320
life = 310
max_death_life = max(death, life)

assert death >= life


def print_common():
    print("unkkey", unkkey)
    print("paddingkey", paddingkey)
    print("seed_num", seed_num)
    print("life", life)
    print("death", death)
    print("max_death_life", max_death_life)
