import os
import torch
import datetime
import pandas as pd
import numpy as np
import torch.nn.functional as F
from openpyxl import Workbook, load_workbook
from create_data import read_runinf,writer_csv

def read_siminf(sim_path): #输入的是一个文件
    key1 = "任务队列大小"
    key2 = "simulation time"
    file = open(sim_path, "r", encoding="utf-8")
    lines = file.readlines()
    taskque = [0]
    time = []
    simtime = []
    walltime = []
    exevent = []
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
            time.append(line_new[1])
            simtime.append(line_new[6])
            walltime.append(int(line_new[10]))
            exevent.append(int(line_new[15]))
    line = lines[-1].split()
    fintime = int(line[2])

    # siminf = [time] + [taskque[1:]] + [simtime] + [walltime] + [exevent]
    siminf = [time] + [simtime] + [walltime]+ [exevent]
    return siminf, fintime

def read_LLC(sim_path):
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


def create_data(file_path,num_thread):
    run_path = os.path.join(file_path, 'pidstat_out_{0}.txt'.format(num_thread))   #rootsim, Graphsim
    llc_path = os.path.join(file_path, 'perf_out_{0}.txt'.format(num_thread))  # rootsim, Graphsim
    # run_path =file_path   #USE
    print(run_path)

    #[[time], [simtime], [walltime], [exevent]]
    # sim_inf, fintime = read_siminf(sim_path)
    # event_num = sim_inf[3][-1]
    # sim_inf = np.array(sim_inf)
    #[[time],[sum_usr],[std_usr],[maxmin_usr],[sum_system],[std_system],[maxmin_system],[sum_wait],[sum_cpu],[std_cpu],[maxmin_cpu],
    # [VSZ],[RSS],[mean_cswch],[std_cswch],maxmin_cswch],[mean_nvcswch],[std_nvcswch],[maxmin_nvcswch]]
    run_inf = read_runinf(num_thread,run_path) #每个线程的平均值
    run_inf = np.array(run_inf)
    # llc_all = read_LLC(llc_path)
    # llc_all = np.array(llc_all)
    # llc = np.mean(llc_all, axis=1)


    #生成数据集
    train_input = run_inf[1:,:]
    # train_output = sim_inf[1:,:]

    # [time],[sum_usr],[std_usr],[maxmin_usr],[sum_system],[std_system],[maxmin_system],[sum_wait],[sum_cpu],[std_cpu],[maxmin_cpu],
    # [VSZ],[RSS],[mean_cswch],[std_cswch],maxmin_cswch],[mean_nvcswch],[std_nvcswch],[maxmin_nvcswch],[taskque]]
    train_input = train_input.astype(float).round(3)
    train_input = train_input.T

    input_overall = np.mean(train_input, 0) #求程序整体的信息，对整个过程求平均
    # [[simtime],[walltime],[exevent]]
    # train_output = np.concatenate((event_num, train_output), axis=0)
    # train_output = train_output.astype(float).round(3)
    # train_output = train_output.T
    # output_overall = np.mean(train_output, 0)
    # print('train_input',train_input.shape)
    # print('train_output',train_output.shape)

    dataset = {}
    dataset['train_input'] = train_input
    # dataset['train_label'] = train_output  # [simtime], [walltime], [exevent]
    dataset['input_overall'] = input_overall ##求程序整体的信息，对整个过程的runinf求平均
    # dataset['output_overall'] = output_overall ###求程序整体的仿真信息，对整个过程的siminf求平均

    # return dataset, fintime, event_num
    return dataset

def writer_allinf(compx, file_path, num_thread, output_path, lp, flag=False, sheet_name='sheet1'):  #ROOTSIM、Graphsim
# def writer_allinf(file_path, num_thread, ENFORCE_LOCALITY, output_path,lp,flag=False,sheet_name='sheet1'):  #USE
    dataset = create_data(file_path, num_thread)
    train_input = dataset['input_overall']
    # [[event_num],[simtime],[walltime],[syntime],[exetime],[exevent]]
    # print(train_input.shape, train_output.shape)

    entity_num = lp
    # [[sum_usr],[std_usr],[maxmin_usr],[sum_system],[std_system],[maxmin_system],[sum_wait],[sum_cpu],[std_cpu],[maxmin_cpu],
    # [VSZ],[RSS],[mean_cswch],[std_cswch],maxmin_cswch],[mean_nvcswch],[std_nvcswch],[maxmin_nvcswch],[taskque]],type=tensor
    # print('sum_usr',train_input[0]),print('std_usr',train_input[1])

    all_inf = pd.DataFrame(
        {'complex':compx,'entity_num': entity_num, 'thread': num_thread, \
         'sum_usr': train_input[0], 'std_usr': train_input[1], 'maxmin_usr': train_input[2],\
         'sum_system': train_input[3], 'std_system': train_input[4], 'maxmin_system': train_input[5],\
         'sum_cpu': train_input[7], 'std_cpu': train_input[8], 'maxmin_cpu': train_input[9], 'VSZ': train_input[10], \
         'RSS': train_input[11], 'mean_cswch': train_input[12], 'std_cswch': train_input[13], \
         'maxmin_cswch': train_input[14], 'mean_nvcswch': train_input[15], \
         'std_nvcswch': train_input[16], 'maxmin_nvcswch': train_input[17],\
         'IO_wr': train_input[18]},index=[0])

    # all_inf = pd.DataFrame(
    #     {'complex':compx,'entity_num': entity_num, 'thread': num_thread, \
    #      'sum_usr': train_input[0],\
    #      'sum_system': train_input[1],\
    #      'sum_cpu': train_input[2], 'VSZ': train_input[3], \
    #      'RSS': train_input[4], 'mean_cswch': train_input[5],\
    #      'mean_nvcswch': train_input[6], \
    #      'IO_wr': train_input[7]},index=[0])

    if flag:
        all_inf, output_path = writer_csv(all_inf, output_path, sheet_name, mode='a')

    return all_inf, output_path, flag

