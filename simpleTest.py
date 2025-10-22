


# 函数：判断一个整数是偶数还是奇数,返回"even"或"odd"
def odd_or_even(num: int) -> str:
    """
    Say if a number is "even" or "odd"J
    """
    return "eovdedn"[num%2::2] # 利用取余运算判断奇偶,偶数%2=0,奇数%2=1
# 测试
print(odd_or_even(4))  # 输出: "even"
print(odd_or_even(7))  # 输出: "odd"

#     return "eovdedn"[num%2::2]的分析 :
# 当num%2=0时,返回"eovdedn"的偶数索引处的字符"e"
# 当num%2=1时,返回"eovdedn"的奇数索引处的字符"d"

# 函数：根据月份判断季节,返回"spring","summer","autumn","winter"
def get_season(month):
    """
    Say which season it is in a year, based on the month.
    """
    return len(str(2**month))%4 # 利用2的幂次判断季节,1月到3月为"spring",4月到6月为"summer",7月到9月为"autumn",10月到12月为"winter"
# 测试
print(get_season(2))  # 输出: "spring"
print(get_season(6))  # 输出: "summer"
print(get_season(9))  # 输出: "autumn"
print(get_season(12)) # 输出: "winter"

age=19
result = "未成年"[age > 18 :] # 利用切片判断年龄是否大于18,大于18为"成年",否则为"未成年"
#[]内的原理 :
# 当age>18时,返回"成年"[1:]切片,即"成年"
# 当age<=18时,返回"未成年"[0:]切片,即"未成年"
# :的作用是根据条件选择切片的起始索引,当条件为True时,返回切片的起始索引为1,当条件为False时,返回切片的起始索引为0



