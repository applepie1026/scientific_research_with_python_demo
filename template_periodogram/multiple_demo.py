import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import time
from multiprocessing import Process, Manager


def test(a, return_list, i):
    # a = {f"{i}": a}
    return return_list.extend(a)


def test2(a, shared_dict, i):
    # a = {f"{i}": a}
    a = {i: a}
    shared_dict.update(a)


# data = {}
# data = np.zeros((2, 10))
# 多线程，并将每线程的执行结束后的返回值按照顺序存入一个列表,共享数组
# 有问题，共享数组，但是数组存入的顺序不可控
# if __name__ == "__main__":
#     for k in range(2):
#         with Manager() as manager:
#             return_list = manager.list()
#             print(return_list)
#             process_list = []
#             for i in range(10):
#                 T1 = time.perf_counter()
#                 print("进程 %s" % i)
#                 a = np.array([i])
#                 p = Process(target=test, args=(a, return_list, i))
#                 p.start()
#                 process_list.append(p)

#             for p in process_list:
#                 p.join()

#             print("All subprocesses done!")
#             print(len(return_list))
#             # 追加保存return_list

#             data[k] = np.array(return_list)
#             # data[f"{i}"] = return_list
# print(data)
# print(data.shape)
# np.savetxt("scientific_research_with_python_demo/data_save/return_list.txt", data, fmt="%s")

# 多线程，并将每线程的执行结束后的返回值按照顺序存入一个列表,共享字典
data = {}


# if __name__ == "__main__":
def mutiple_demo():
    for k in range(2):
        with Manager() as manager:
            shared_dict = manager.dict()
            print(shared_dict)
            process_list = []
            for i in range(10):
                T1 = time.perf_counter()
                print("进程 %s" % i)
                a = np.array([i, i + k])
                p = Process(target=test2, args=(a, shared_dict, i))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()

            print("All subprocesses done!")
            print(len(shared_dict))
            # print(shared_dict)
            # 将shared_dict存入data
            data[f"{k}"] = shared_dict.copy()
    return data


# print(data)
# print(data["0"])
# print(data["0"][2])


def data_collect(data, data_length, process_num_all, test_length):
    collected_data = np.zeros((test_length, data_length))
    for j in range(test_length):
        # 将process_num_all个数组合并
        data_list = []
        for i in range(process_num_all):
            data_list.append(data[f"{j}"][i])
        collected_data[j, :] = np.concatenate(data_list, axis=0)
    return collected_data


data = mutiple_demo()
data_array = data_collect(data, 20, 10, 2)
print(data_array)
# np.savetxt("scientific_research_with_python_demo/data_save/mutiple_demo.txt", data_array, delimiter=",")
# data[k] = np.array(return_list)
# data[f"{i}"] = return_list
