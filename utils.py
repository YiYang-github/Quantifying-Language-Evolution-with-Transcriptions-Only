from concurrent.futures import ThreadPoolExecutor
import numpy as np
import re

import matplotlib.pyplot as plt
from collections import Counter

from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # 地球平均半径，单位为公里

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat / 2)**2 + cos(lat1) * cos(lat2) * sin(dLon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def plot_length_distribution(data, title_text = 'Pie Chart of Lengths'):
    # 计算长度的分布
    length_counts = Counter(data)
    sizes = length_counts.values()  # 各部分大小
    labels = length_counts.keys()   # 各部分标签
    
    # 定义颜色
    colors = plt.cm.Paired(range(len(labels)))
    
    # 创建饼图
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
            wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 12})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.show()

def calculate_zero_fraction(arr):
    """
    Calculate the fraction of zero elements in a 2D NumPy array.

    Parameters:
        arr (np.ndarray): The 2D array to analyze.

    Returns:
        float: The fraction of elements that are zero in the array.
    """
    total_elements = arr.size  # 获取数组中的总元素数
    zero_count = np.count_nonzero(arr == 0)  # 计算数组中零的数量
    zero_fraction = zero_count / total_elements  # 计算零元素的比例
    return zero_fraction


def Levenshtein_distance(list1, list2, transition_matrix):
    """
    A General Extension to Levenshtein Distance (Edit Distance).
    The cost from list1 to list2.

    Input:
        transition_matrix: np.array [n+1, n+1]; (i, j) represents the cost transferring i to j.
                           The transition_matrix for edit distance is np.ones((n+1, n+1)) - np.identity(n+1)

        list1, list2: the elements are from 1 to n (1-based indexing)

    Output:
        returns the segment x segment frequency matrix and the overall cost
    """

    len1, len2 = len(list1), len(list2)
    dp = np.zeros((len1 + 1, len2 + 1))
    count_joint_p = np.zeros((transition_matrix.shape[0], transition_matrix.shape[0]))

    # Initialize dp array for deletions and insertions
    for i in range(1, len1 + 1):
        dp[i][0] = dp[i-1][0] + transition_matrix[list1[i-1], 0] 

    for j in range(1, len2 + 1):
        dp[0][j] = dp[0][j-1] + transition_matrix[0, list2[j-1]]
    
    # Dynamic programming to solve subproblems
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            replace_cost = dp[i-1][j-1] + transition_matrix[list1[i-1], list2[j-1]]
            insert_cost = dp[i][j-1] + transition_matrix[0, list2[j-1]]
            delete_cost = dp[i-1][j] + transition_matrix[list1[i-1], 0]

            # Choose the minimum cost
            dp[i][j] = min(replace_cost, insert_cost, delete_cost)

            # Update the count_joint_p based on the chosen operation
            if dp[i][j] == replace_cost:
                count_joint_p[list1[i-1], list2[j-1]] += 1
            elif dp[i][j] == insert_cost:
                count_joint_p[0, list2[j-1]] += 1
            elif dp[i][j] == delete_cost:
                count_joint_p[list1[i-1], 0] += 1
    
    return dp[len1][len2], count_joint_p


def sparse_phonemes(dialects_speeches, phoneme2num, special_characters):
    """
    解析 dialects_speeches List[List[str]] -> List[List[List[int]]], 借助phonemes2num Dict[str, int] 把每个str convert 为 str
    [] 表示无法解析的字符串

    Args:
    - dialects_speeches List[List[str]]: t ɔ 55 format like this
    - phoneme2num: dict [str, int],  只有音素在phoneme2num的元素会被保留
    - special_characters: dict，如果出现special_characters，该元素会被立即丢弃
     
    Returns:
    - initials_lists: List[List[List[[int]]], 内层 List[[int] 表示每个initials构成的音素
    - finals_lists:   List[List[List[[int]]], 内层 List[[int] 表示每个finals构成的因素
    - tones_lists:    List[List[List[[int]]], 内层 List[[int] 表示每个声调五度制转写
    - all_lists:      List[List[List[[int]]], 内层 List[[int] 表示每个initials, finals构成的音素
    """

    initials_lists, finals_lists, all_lists, tones_lists = [], [], [], []
    for dialect_list in  dialects_speeches:

        initials_list, finals_list, alls_list, tones_list = [], [], [], []

        for chars in dialect_list:

            if check_format(chars, phoneme2num, special_characters):
                all_list, initial_list, final_list, tone_list = check_format(chars, phoneme2num, special_characters)
                initials_list.append(initial_list)
                finals_list.append(final_list)
                alls_list.append(all_list)
                tones_list.append(tone_list) 
            
            else:
                initials_list.append([])
                finals_list.append([])
                alls_list.append([])
                tones_list.append([]) 

        initials_lists.append(initials_list)
        finals_lists.append(finals_list)
        all_lists.append(alls_list)
        tones_lists.append(tones_list) 
        
    return initials_lists, finals_lists, all_lists, tones_lists

def check_format(chars, phoneme2num, special_characters):
    """
    检查是否符合format，符合则返回正确格式

    Args:
    - chars: str
    - phoneme2num: dict [str, int],  只有音素在phoneme2num的元素会被保留
    - special_characters: dict，如果出现special_characters，该元素会被立即丢弃

    Returns:
        if format is not correct, return false.
        otherwise, return all_list, initial_list, final_list, tone_list
    """

    all_list, initial_list, final_list, tone_list = [], [], [], []

    if len(chars.split()) != 3:
            return False

    for char in special_characters:
        if char in chars:
            return False
    
    for initials in chars.split()[0]:
        for initial in initials:
            if initial in phoneme2num.keys():
                initial_list.append(phoneme2num[initial])
                all_list.append(phoneme2num[initial])
            else:
                return False
    
    for finals in chars.split()[1]:
        for final in finals:
            if final in phoneme2num.keys():
                final_list.append(phoneme2num[final])
                all_list.append(phoneme2num[final])
            else:
                return False
                        
    if re.fullmatch(r'[1-5]{1,3}', chars.split()[2]):
        tone_list = [int(char) for char in chars.split()[2]]
    else:
        return False

    return all_list, initial_list, final_list, tone_list

def sparsity(phoneme_list):
    """
    Calculate sparsity
    """
    total, _sparsity = 0, 0
    for _list in phoneme_list:
        for ele in _list:
            if ele:
                _sparsity += 1
            total += 1

    print(f"The data sparsity is {_sparsity / (total):.6f}.")