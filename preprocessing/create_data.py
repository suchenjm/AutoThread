import os
import torch
import datetime
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from sklearn.model_selection import train_test_split
import bisect


def truncate_to_min_length(arr):
    # 获取所有子列表的长度，并找到最小值
    lengths = [len(sub_arr) for sub_arr in arr]
    min_length = min(lengths)
    # 对每个子列表进行截断
    return [sub_arr[:min_length] for sub_arr in arr]

def create_data(file_path,num_thread):
    run_path = os.path.join(file_path, 'pidstat_out_{0}.txt'.format(num_thread))
    sim_path = os.path.join(file_path, 'simulation_out_{0}.txt'.format(num_thread))

    #[[time],[taskque],[simtime],[walltime],[syntime],[exetime],[exevent]]
    print(sim_path)
    try:
        sim_inf, excute1_time, excute2_time, create_time, fintime,_ = read_siminf(num_thread,sim_path)
        event_num = sim_inf[6][-1]
        sim_inf = time_align(sim_inf, 1)
    except:
        sim_inf, excute1_time, excute2_time, create_time, fintime = [[]],[],[],[],[]
        event_num = []

    #[[time],[sum_usr],[std_usr],[maxmin_usr],[sum_system],[std_system],[maxmin_system],[sum_wait],[sum_cpu],[std_cpu],[maxmin_cpu],
    # [VSZ],[RSS],[mean_cswch],[std_cswch],maxmin_cswch],[mean_nvcswch],[std_nvcswch],[maxmin_nvcswch]]
    run_inf = read_runinf(num_thread,run_path) #每个线程的平均值
    run_inf = np.array(run_inf)

    result = align_timeseries(sim_inf, run_inf)

    # #生成数据集
    train_input = result[8:, :]
    train_output = result[1:7, :]

    # [time],[sum_usr],[std_usr],[maxmin_usr],[sum_system],[std_system],[maxmin_system],[sum_wait],[sum_cpu],[std_cpu],[maxmin_cpu],
    # [VSZ],[RSS],[mean_cswch],[std_cswch],maxmin_cswch],[mean_nvcswch],[std_nvcswch],[maxmin_nvcswch],[taskque]]
    # train_input = np.concatenate((train_input, taskque), axis=0) #无time，记录程序每一秒的信息
    train_input = train_input.astype(float).round(3)
    train_input = torch.from_numpy(train_input.T)

    input_overall = torch.nanmean(train_input, dim=0) #求程序整体的信息，对整个过程求平均
    # [[event_num],[simtime],[walltime],[syntime],[exetime],[exevent]]
    event_num = torch.ones(1,len(train_input[:,0]))*event_num
    train_output = np.concatenate((event_num, train_output), axis=0)
    train_output = train_output.astype(float).round(3)
    train_output = torch.from_numpy(train_output.T)
    output_overall = torch.nanmean(train_output, dim=0)
    # print('train_input',train_input.shape)
    # print('train_output',train_output.shape)

    dataset = {}
    dataset['train_input'] = train_input
    dataset['train_label'] = train_output
    dataset['input_overall'] = input_overall
    dataset['output_overall'] = output_overall

    return dataset, fintime

def read_runinf(num_thread,run_path): #输入的是一个文件
    key = "Time"
    flag = int(num_thread)
    file = open(run_path, "r",encoding="utf-8")
    lines = file.readlines()
    time = []
    VSZ = []
    RSS = []
    Gid_usr = []
    Gid_system = []
    Gid_CPU = []
    usr = [[] for i in range(flag + 1)] #[线程数 ，运行时间]——每个子线程的运行信息
    system = [[] for i in range(flag + 1)]
    wait = [[] for i in range(flag + 1)]
    CPU = [[] for i in range(flag + 1)]
    io_wr = [[] for i in range(flag + 1)]
    cswch = [[] for i in range(flag + 1)]
    nvcswch = [[] for i in range(flag + 1)]  # 第一个是主线程，其余是工作线程
    for i, line in enumerate(lines, start=0):  # enumerate按序号，值的形式输出，序号从start开始
        if (key in line):
            # print("lines=",lines[i+1])
            # print("flag=",flag)
            # print("lines=",len(lines[i+flag+2]))
            if (i + flag + 2 < len(lines)) and (len(lines[i + flag + 2]) < 2):  #没有记录到进程信息
                try:
                    lines[i + 1] = lines[i + 1].split()
                except:
                    continue

                Gid_usr.append(lines[i+1][4])
                Gid_system.append(lines[i+1][5])
                Gid_CPU.append(lines[i+1][8])

                # print("lines = ",lines[i+1])
                time.append(lines[i + 1][0])
                VSZ.append(int(lines[i + 1][12]))  # 进程使用虚拟内存大小，KB
                RSS.append(int(lines[i + 1][13]))  # 进程使用物理内存大小，KB
                usr[0].append(float(lines[i + 1][4]))
                system[0].append(float(lines[i + 1][5]))
                wait[0].append(float(lines[i + 1][7]))
                CPU[0].append(float(lines[i + 1][8]))
                io_wr.append(float(lines[i + 1][16]))
                cswch[0].append(float(lines[i + 1][19]))  # 每秒自愿上下文切换次数
                nvcswch[0].append(float(lines[i + 1][20]))  # 每秒非自愿上下文切换次数
                for thrads in range(flag):
                    try:
                        lines[i + 2 + thrads] = lines[i + 2 + thrads].split()
                        usr[thrads + 1].append(float(lines[i + 2 + thrads][4]))
                        system[thrads + 1].append(float(lines[i + 2 + thrads][5]))
                        wait[thrads + 1].append(float(lines[i + 2 + thrads][7]))
                        CPU[thrads + 1].append(float(lines[i + 2 + thrads][8]))
                        cswch[thrads + 1].append(float(lines[i + 2 + thrads][19]))
                        nvcswch[thrads + 1].append(float(lines[i + 2 + thrads][20]))
                    except:
                        continue

            elif (i + flag + 3 <= len(lines)):  #记录到进程信息
                try:
                    lines[i + 1] = lines[i + 1].split()
                except:
                    continue
                # print("lines = ",lines[i+1])
                Gid_usr.append(lines[i+1][4])
                Gid_system.append(lines[i+1][5])
                Gid_CPU.append(lines[i+1][8])
                time.append(lines[i + 1][0])
                VSZ.append(int(lines[i + 1][12]))  # 进程使用虚拟内存大小，KB
                RSS.append(int(lines[i + 1][13]))  # 进程使用物理内存大小，KB
                lines[i + 2] = lines[i + 2].split()
                usr[0].append(float(lines[i + 2][4]))
                system[0].append(float(lines[i + 2][5]))
                wait[0].append(float(lines[i + 2][7]))
                CPU[0].append(float(lines[i + 2][8]))
                io_wr.append(float(lines[i + 2][16]))
                cswch[0].append(float(lines[i + 2][19]))  # 每秒自愿上下文切换次数
                nvcswch[0].append(float(lines[i + 2][20]))  # 每秒非自愿上下文切换次数
                # print("lines=",lines[i+1])
                # print("flag=",flag)
                for thrads in range(flag):
                    try:
                        lines[i + 3 + thrads] = lines[i + 3 + thrads].split()
                        usr[thrads + 1].append(float(lines[i + 3 + thrads][4]))
                        system[thrads + 1].append(float(lines[i + 3 + thrads][5]))
                        wait[thrads + 1].append(float(lines[i + 3 + thrads][7]))
                        CPU[thrads + 1].append(float(lines[i + 3 + thrads][8]))
                        cswch[thrads + 1].append(float(lines[i + 3 + thrads][19]))
                        nvcswch[thrads + 1].append(float(lines[i + 3 + thrads][20]))
                    except:
                        continue

    usr = truncate_to_min_length(usr)
    system = truncate_to_min_length(system)
    wait = truncate_to_min_length(wait)
    CPU = truncate_to_min_length(CPU)
    cswch = truncate_to_min_length(cswch)
    nvcswch = truncate_to_min_length(nvcswch)
    VSZ = VSZ[0:len(usr[0])]
    RSS = RSS[0:len(usr[0])]
    io_wr = io_wr[-len(usr[0]):]
    # Gid_usr = Gid_usr[0:len(usr[0])]
    # Gid_system = Gid_system[0:len(usr[0])]
    # Gid_CPU = Gid_CPU[0:len(usr[0])]
    time = time[0:len(usr[0])]
    sum_usr = np.sum(usr, axis=0).tolist()
    std_usr = np.std(usr, axis=0).tolist()
    maxmin_usr = (np.max(usr, axis=0) - np.min(usr, axis=0)).tolist()
    sum_system = np.sum(system, axis=0).tolist()
    std_system = np.std(system, axis=0).tolist()
    maxmin_system = (np.max(system, axis=0) - np.min(system, axis=0)).tolist()
    sum_wait = np.sum(wait, axis=0).tolist()
    sum_cpu = np.sum(CPU, axis=0).tolist()
    std_cpu = np.std(CPU, axis=0).tolist()
    maxmin_cpu = (np.max(CPU, axis=0) - np.min(CPU, axis=0)).tolist()
    mean_cswch = np.mean(cswch, axis=0).tolist()
    std_cswch = np.std(cswch, axis=0).tolist()
    maxmin_cswch = (np.max(cswch, axis=0) - np.min(cswch, axis=0)).tolist()
    mean_nvcswch = np.mean(nvcswch, axis=0).tolist()
    std_nvcswch = np.std(nvcswch, axis=0).tolist()
    maxmin_nvcswch = (np.max(nvcswch, axis=0) - np.min(nvcswch, axis=0)).tolist()
    # runinf = [time] + [sum_usr] + [std_usr] + [sum_system] + [std_system] + [sum_cpu] + [std_cpu] + [VSZ] + [RSS] + [mean_cswch] + [std_cswch] + [mean_nvcswch] + [std_nvcswch]
    # runinf = [time] + [sum_usr] + [std_usr] + [maxmin_usr] + [sum_system] + [std_system] + [maxmin_system] + [sum_wait] + [sum_cpu] + [std_cpu] + [maxmin_cpu] + [VSZ] + [RSS] + [mean_cswch] + [std_cswch] + [maxmin_cswch] + [mean_nvcswch] + [std_nvcswch] + [maxmin_nvcswch] + [io_wr]
    # return runinf
    processinf = [time] + [Gid_usr] + [std_usr] + [maxmin_usr] + [Gid_system] + [std_system] + [maxmin_system] + [sum_wait] + [Gid_CPU] + [std_cpu] + [maxmin_cpu] + [VSZ] + [RSS] + [cswch[0]] + [std_cswch] + [maxmin_cswch] + [nvcswch[0]] + [std_nvcswch] + [maxmin_nvcswch] + [io_wr]
    # # processinf = [time] + [Gid_usr] + [Gid_system] + [sum_wait] + [Gid_CPU] + [VSZ] + [RSS] + [cswch[0]] + [nvcswch[0]] + [io_wr]
    return processinf
def read_siminf(num_thread,sim_path): #输入的是一个文件
    key1 = "任务队列大小"
    key2 = "simulation time"
    key3 = "event1 excute time"
    key4 = "event2 excute time"
    key5 = "thread time"
    key6 = "当前线程数"
    flag = int(num_thread)
    file = open(sim_path, "r", encoding="utf-8")
    lines = file.readlines()
    taskque = [0]
    time = []
    simtime = []
    walltime = []
    syntime = []
    exetime = []
    exevent = []
    create_time = []
    excute1_time = []
    excute2_time = []
    thread_queue = []
    for i, line in enumerate(lines, start=1):
        if key1 in line:
            line = line.replace("-", ":")
            line = line.split(":")
            if len(line) > 2:
                if line[1] == "2024" and int(lines[i][:4]) != 2024:  #数值在单独一行
                    line_new = lines[i].split("2024")
                    # print(line_new)
                    taskque.append(int(line_new[0]))
                elif int(line[1]) > 10000 or int(line[1][0]) == 0:
                    taskque.append(int(line[1][:-4]))
                elif len(lines[i+1]) < 5:
                    taskque.append(int(lines[i+1]))
                else:
                    last = taskque[-1]
                    taskque.append(last)
            else:
                taskque.append(int(line[1]))
        elif key2 in line:
            line_new = line.replace("=", ",")
            line_new = line_new.replace(",", " ")
            line_new = line_new.replace("\n", " ")
            line_new = line_new.split(" ")
            # print(line_new)
            # time.append(line_new[1])
            # simtime.append(line_new[6])
            # walltime.append(int(line_new[10]))
            # syntime.append(int(line_new[15]))
            # exetime.append(int(line_new[20]))
            # exevent.append(int(line_new[25]))
            ##2025-06-04
            # time.append(line_new[1])
            # simtime.append(line_new[6])
            # walltime.append(int(line_new[10]))
            # syntime.append(int(line_new[22]))
            # exetime.append(int(line_new[16]))
            # exevent.append(int(line_new[27]))

            time.append(line_new[-28])
            simtime.append(line_new[-23])
            walltime.append(int(line_new[-19]))
            syntime.append(int(line_new[-7]))
            exetime.append(int(line_new[-13]))
            exevent.append(int(line_new[-2]))
        elif key3 in line:
            line_new = line.replace("=", " ")
            line_new = line_new.replace("\n", "")
            line_new = line_new.split(" ")
            # print('line_new =',line_new)
            excute1_time.append(int(line_new[7]))  # 计算事件1平均执行时间
        elif key4 in line:
            line_new = line.replace("=", " ")
            line_new = line_new.replace("\n", "")
            line_new = line_new.split(" ")
            # print('line_new =',line_new)
            try:
                excute2_time.append(int(line_new[7]))  # 计算事件2平均执行时间,原本是7
            except:
                pass
            # excute2_time.append(int(line_new[-2]))  # 计算事件2平均执行时间
        elif key5 in line:
            line_new = line.replace("=", " ")
            line_new = line_new.replace("\n", "")
            line_new = line_new.split(" ")
            # print('line_new =',line_new)
            create_time = int(line_new[8]) / int(line_new[5])  # 创造线程平均执行时间
        elif key6 in line:
            line_new = line.replace("=", " ")
            line_new = line_new.replace("\n", "")
            line_new = line_new.split(" ")
            # print('line_new =',line_new)
            thread_queue.append(int(line_new[1]))

    try:
        line = lines[-1].split()
        fintime = int(line[2])
    except:
        fintime = walltime[-1]

    siminf = [time] + [taskque[1:]] + [simtime] + [walltime] + [syntime] + [exetime] + [exevent]
    return siminf, excute1_time, excute2_time, create_time, fintime, thread_queue

def read_LLC(sim_path):
    # key1 = "cache-references"
    # key2 = "cache-misses"
    key1 = "cache-references"
    key2 = "cache-misses"
    file = open(sim_path, "r", encoding="utf-8")
    lines = file.readlines()
    load = []
    store = []
    for i, line in enumerate(lines, start=1):
        if key1 in line:
            line_new = line.split()
            load.append(int(line_new[1].replace(',', '')))
        elif key2 in line:
            line_new = line.split()
            store.append(int(line_new[1].replace(',', '')))
    llc = [load] + [store]
    return llc

def length_align(sim_inf1):
    min_length = min(len(sublist) for sublist in sim_inf1)
    sim_inf = []
    for sublist in sim_inf1:
        # 如果当前子列表长度大于最短长度，保留最后min_length个元素
        if len(sublist) > min_length:
            sim_inf.append(sublist[-min_length:])  # 对齐所有子列长度
        else:
            # 如果当前子列表长度已经是最短长度，则直接保留
            sim_inf.append(sublist)
    return sim_inf
def time_align(sim_inf1,delta_t=1):
    sim_inf = length_align(sim_inf1)
    flag = 0
    sim_inf = np.array(sim_inf)
    start_time = datetime.datetime.strptime(sim_inf[0,0], "%H:%M:%S.%f")
    if delta_t==1:
        range1 = start_time.replace(microsecond=0)
        range2 = range1 + datetime.timedelta(seconds=1)
    else:
        st = int(delta_t * 1000000)
        range0 = start_time.replace(microsecond=st)
        if range0 < start_time:
            range1 = range0
        else:
            range1 = start_time.replace(microsecond=0)
        range2 = range1 + datetime.timedelta(microseconds=st)
    # print(start_time)
    inf = []
    inf_mean = []
    timestap = [[range1.strftime("%H:%M:%S.%f")]]
    for i in range(len(sim_inf[0])):
        time = datetime.datetime.strptime(sim_inf[0,i], "%H:%M:%S.%f")
        # time = time.strftime("%H:%M:%S.%f")[:-3]
        if (time<range2 and time>=range1):
            if sim_inf[1,i] is not None:
                a = sim_inf[1:,i].tolist()
                inf.append(list(map(float, a)))
            if i < len(sim_inf[0])-1:
                continue
        inf_mean.append(np.mean(inf,axis=0).round(3))
        range1 = range1 + datetime.timedelta(seconds=delta_t)
        range2 = range2 + datetime.timedelta(seconds=delta_t)
        if sim_inf[1, i] is not None:
            inf = sim_inf[1:,i].tolist()
            inf = [list(map(float, inf))]
        timestap.append([range1.strftime("%H:%M:%S.%f")])
    inf_mean = np.array(inf_mean)
    timestap = np.array(timestap[:-1])
    sim_inf1 = np.concatenate((timestap,inf_mean),axis=1)
    return sim_inf1.T

def align_timeseries(arr1, arr2,time_tolerance=0.5):
    lon1,len1 = arr1.shape
    lon2, len2 = arr2.shape
    times1 = arr1[0]
    times2 = arr2[0]
    # 转换时间格式为datetime对象
    def convert_time1(t_str):
        return datetime.datetime.strptime(t_str, "%H:%M:%S.%f")

    def convert_time(t_str):
        if "秒" == t_str[-1]:
            return datetime.datetime.strptime(t_str.replace("时", ":").replace("分", ":").replace("秒", ""),"%H:%M:%S")
        else:
            return datetime.datetime.strptime(t_str, "%H:%M:%S.%f")
    # 转换所有时间戳
    timestamps1 = np.array([convert_time(t) for t in times1])
    timestamps2 = np.array([convert_time(t) for t in times2])

    # 创建结果容器 (25行 x 80列)
    result = np.full((lon1 + lon2, len1), np.nan, dtype=object)
    # 对齐时间并填充数据
    for col_idx in range(len1):  # 遍历arr1的80列
        t1 = timestamps1[col_idx]
        # 在arr2中寻找匹配的时间（允许1秒误差）
        matches = np.where(np.abs((timestamps2 - t1).astype('timedelta64[s]').astype(float)) <= time_tolerance)[0]
        if matches.size > 0:
            # 取第一个匹配项
            match_idx = matches[0]
            # 填充arr1的所有17行数据
            result[:lon1, col_idx] = arr1[:, col_idx]
            # 填充arr2的7行数据（跳过时间行）
            result[lon1:lon1 + lon2, col_idx] = arr2[:, match_idx]
        else:
            # 无匹配时只填充arr1的数据
            result[:lon1, col_idx] = arr1[:, col_idx]
    return result


def dataset_join(pathlist,threadlist,device='cpu'):
    num = len(pathlist)
    data = create_data(pathlist[0], threadlist[0])
    input = data['train_input']
    output = data['train_label']
    for i in range(1,num):
        data = create_data(pathlist[i],threadlist[i])
        input = torch.cat((input,data['train_input']),0)
        output = torch.cat((output,data['train_label']),0)

    #归一化
    # input = F.normalize(input,dim=0)
    # output = F.normalize(output,dim=0)
    #划分测试集与验证集
    x_train,x_test,y_train,y_test = train_test_split(input,output,test_size=0.3,random_state=40)

    dataset = {}
    dataset['train_input'] = x_train.to(device)
    dataset['train_label'] = y_train.to(device)
    dataset['test_input'] = x_test.to(device)
    dataset['test_label'] = y_test.to(device)

    return dataset

def writer_csv(all_inf,output_path,sheet_name,mode):
    if not os.path.exists(output_path):
        wb = Workbook()
        wb.save(output_path)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            all_inf.to_excel(writer, sheet_name=sheet_name, index=False)
    elif mode=='a':
        df = pd.read_excel(output_path, engine='openpyxl', sheet_name=sheet_name)
        row = df.shape[0]
        all_inf = pd.concat([pd.DataFrame(columns=df.columns), all_inf], ignore_index=True)
        book = load_workbook(output_path)
        for sheet in book.sheetnames:
            ws = book[sheet]
            ws.sheet_state = 'visible'  # 将所有工作表设置为可见
        with pd.ExcelWriter(output_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # 追加新数据,追加前必须先格式化新数据,否则新数据缺少列,或是列顺序不对会导致数据紊乱
            all_inf.to_excel(writer, sheet_name=sheet_name, startrow=row + 1, index=False, header=False)
    else:
        with pd.ExcelWriter(output_path, engine='openpyxl',mode='a', if_sheet_exists='replace') as writer:
            all_inf.to_excel(writer, sheet_name=sheet_name, index=False)


    return all_inf,output_path

def read_trace(file_folder,output_path,lp,complex,mothed):  # file_folder = E:\paper\论文\论文1-线程预测\数据\AMD_PCS\Airline\nohot_data_2000\scenario5
    files1 = os.listdir(file_folder)  # scenario文件夹下
    files1.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))
    flag = 0
    for filetxt in files1:
        num_thread = int(filetxt.split(".")[0].split("_")[2])  # "E:\paper\论文\论文1-线程预测\数据\X86_PCS\PCS\complex_5000\lp_3600_tm_1000\simulation_out_50.txt"
        if filetxt[0] == "s":
            flag += 1
            sim_path = os.path.join(file_folder, filetxt) # "E:\paper\论文\论文1-线程预测\数据\AMD_PCS\Airline\nohot_data_2000\scenario5\simulation_out_80.txt"
            print(sim_path)
            # try:
            siminf, excute1_time, excute2_time, create_time, fintime = read_siminf(num_thread, sim_path)
            # siminf = time_align(siminf)
            # except:
            #     pass
        elif filetxt[:2] == "pi":
            flag += 1
            # run_path = os.path.join(file_folder, filetxt)
            # run_inf1 = read_runinf(num_thread, run_path)  # 每个线程的平均值
            # run_inf = length_align(run_inf1)
            # run_inf = np.array(run_inf)
        elif filetxt[:2] == "pe":
            flag += 1
            # llc_path = os.path.join(file_folder, filetxt)
            # llc_inf = read_LLC(llc_path)  # 每个线程的平均值
            # llc_inf = np.array(llc_inf)
        if flag % 3 ==0:
            # result = align_timeseries(siminf, run_inf)
            # if len(result[0]) >= len(llc_inf[0]):
            #     result = np.concatenate((result[:, 0:len(llc_inf[0, :])], llc_inf), axis=0)
            # else:
            #     result = np.concatenate((result, llc_inf[:,0:len(result[0,:])]), axis=0)
            # # result = np.concatenate((result[:,0:len(llc_inf[0,:])],llc_inf),axis=0)
            # allinframe = pd.DataFrame(
            #     {'round':round,'mothed':mothed,'complex':complex,'entity num':lp,'thread':num_thread,'time': result[0], 'task queue': result[1], 'simulation time': result[2], 'wall time': result[3],\
            #      'synchronize time': result[4],'excute time': result[5], 'executed events': result[6],'time2':result[7],'process_usr': result[8], 'std_usr': result[9],\
            #      'maxmin_usr': result[10], 'sum_cpu': result[15], 'std_cpu': result[16], 'maxmin_cpu': result[17], 'VSZ': result[18], 'RSS': result[19], 'mean_cswch': result[20],\
            #      'std_cswch': result[21], 'maxmin_cswch': result[22], 'mean_nvcswch': result[23], \
            #     'std_nvcswch': result[24], 'maxmin_nvcswch': result[25],'cache-references':result[26],'cache-misses':result[27]})

            allinframe = pd.DataFrame(
                    {'mothed':mothed,'complex':complex,'entity num':lp,'thread':num_thread,'excute2_time':excute2_time,'exetime':siminf[5][1:],'syntime':siminf[4][1:],'wall_time':siminf[3][1:]})

            # if flag:
            all_inf, output_path = writer_csv(allinframe, output_path, "Sheet", mode='a')


#-------------------------------------------图：单线程仿真负载与仿真时间的关系-----------------------------------------------------
Ar1c100lp4_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_Airline\\round_1\Static_complex_1000_0.4_5_2\scenario4_VarEntiy\simulation_out_1.txt")
Ar3c100lp4_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_Airline\\round_3\Static_complex_1000_0.4_5_2\scenario4_VarEntiy\simulation_out_1.txt")
Ar5c100lp4_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_Airline\\round_5\Static_complex_1000_0.4_5_2\scenario4_VarEntiy\simulation_out_1.txt")
Ar3c100lp5_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_Airline\\round_3\Static_complex_1000_0.4_5_2\scenario5_VarEntiy\simulation_out_1.txt")
Ar5c200lp5_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_Airline\\round_5\Static_complex_2000_0.4_5_2\scenario5_VarEntiy\simulation_out_1.txt")

Ar1c100lp4_exetime = np.array(Ar1c100lp4_siminf[-2])/1000.0
Ar3c100lp4_exetime = np.array(Ar3c100lp4_siminf[-2])/1000.0
Ar5c100lp4_exetime = np.array(Ar5c100lp4_siminf[-2])/1000.0
Ar3c100lp5_exetime = np.array(Ar3c100lp5_siminf[-2])/1000.0
Ar5c200lp5_exetime = np.array(Ar5c200lp5_siminf[-2])/1000.0
plt.figure()
plt.plot(Ar1c100lp4_exetime,label='roh=1,t=100,n=800')
plt.plot(Ar3c100lp4_exetime,label='roh=3,t=100,n=800')
plt.plot(Ar5c100lp4_exetime,label='roh=5,t=100,n=800')
plt.plot(Ar3c100lp5_exetime,label='roh=3,t=100,n=1261')
plt.plot(Ar5c200lp5_exetime,label='roh=5,t=200,n=1261')
plt.ylabel("Excute time per step(ms)")
plt.xlabel("simulation step")
plt.xlim([0,1000])
plt.title("UVA load distribution")
plt.legend()
plt.grid(True,axis='y')
plt.savefig("C:\\Users\\JmSu\Desktop\\UAV.svg", dpi=300)
plt.show()

Pr1c50lp36_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_PCS\\round_1\static_complex_500\lp_2500_tm_2000\simulation_out_1.txt")
Pr3c50lp36_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_PCS\\round_3\static_complex_500\lp_3600_tm_2000\simulation_out_1.txt")
Pr5c50lp36_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_PCS\\round_5\static_complex_500\lp_3600_tm_2000\simulation_out_1.txt")
Pr5c50lp16_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_PCS\\round_5\static_complex_500\lp_1600_tm_2000\simulation_out_1.txt")
Pr5c100lp16_siminf, _, Ar2c100lp5_excute2_time, _, _,Ar2c100lp5_thread = read_siminf(1, "E:\paper\论文\论文1-线程预测\数据\Dynamic\AMD_PCS\\round_5\static_complex_1000\lp_1600_tm_2000\simulation_out_1.txt")


Pr1c50lp36_exetime = np.array(Pr1c50lp36_siminf[-2])/1000.0
Pr3c50lp36_exetime = np.array(Pr3c50lp36_siminf[-2])/1000.0
Pr5c50lp36_exetime = np.array(Pr5c50lp36_siminf[-2])/1000.0
Pr5c50lp16_exetime = np.array(Pr5c50lp16_siminf[-2])/1000.0
Pr5c100lp16_exetime = np.array(Pr5c100lp16_siminf[-2])/1000.0
fig,ax=plt.subplots()
cycle=10
ax.plot(Pr1c50lp36_exetime[::cycle],label='roh=1,t=50,n=2500')
ax.plot(Pr3c50lp36_exetime[::cycle],label='roh=3,t=50,n=3600')
ax.plot(Pr5c50lp36_exetime[::cycle],label='roh=5,t=50,n=3600')
ax.plot(Pr5c50lp16_exetime[::cycle],label='roh=5,t=50,n=1600')
ax.plot(Pr5c100lp16_exetime[::cycle],label='roh=5,t=100,n=1600')
plt.title("PCS load distribution")
ax.set_ylabel("Excute time per step(ms)")
plt.xlabel("simulation step")
ax.legend(loc="best")
plt.grid(True,axis='y')
plt.savefig("C:\\Users\\JmSu\Desktop\\PCS.svg", dpi=300)
plt.show()


# _,_,run_inf1 = read_runinf(8, "xxxx\scenario5_staticEntiy\simulation_out_10.txt")  # 每个线程的平均值
# run_inf = length_align(run_inf1)
# run_inf = np.array(run_inf)
# print(run_inf)
# read_siminf(1, "xxxx\simulation_out_1.txt")
# read_siminf(1, "xxxx\simulation_out_10.txt")



