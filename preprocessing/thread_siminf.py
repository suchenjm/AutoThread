import os
from pylab import *
import pandas as pd
import numpy as np
from create_dataset import read_siminf,read_newversion_siminf
from openpyxl import Workbook, load_workbook
# from xlsxwriter import Workbook


def thread_siminf(file_folder,output_path,complex,type):
    files = os.listdir(file_folder)  # 此文件夹下有几个场景  (xxxxxxxx\step0.1-0.8_time2000\data3000_0.2)
    files.sort()
    Entity = {'scenario2':206,'scenario3':400,'scenario4':800,'scenario5':1261,'scenario6':1426}
    optimal_thread = 0
    for file in files:
        thread_num = []
        fintime = []
        mean_queue = []
        mean_excute1_time = []
        mean_excute2_time = []
        mean_synchronize_time = []
        mean_excute_time = []
        mean_create_time = []
        sum_synchronize_time = []
        sum_excute_time = []
        sum_create_time = []
        event_num = []
        # entity_num = Entity[file]
        entity_num = int(file.split("_")[1])
        file_path = os.path.join(file_folder,file)  # (xxxxxxxx\step0.1-0.8_time2000\data3000_0.2\scenario2)
        files1 = os.listdir(file_path)  # scenario文件夹下
        files1.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))
        flag = 0
        start_thread = files1[0].split(".")[0].split("_")[2]
        min_time = 1000000
        fintime1 = 1000000
        for filetxt in files1:
            num_thread = int(filetxt.split(".")[0].split("_")[2])  # 工作线程数
            if filetxt[0] == "s":
                flag += 1
                sim_path = os.path.join(file_path, filetxt)
                print(sim_path)
                try:
                    siminf,excute1_time,excute2_time,create_time,fintime1 = read_siminf(num_thread, sim_path)
                except:
                    continue

                # mean_excute1_time.append(np.mean(excute1_time))
                mean_excute2_time.append(np.mean(excute2_time))
                mean_synchronize_time.append(np.mean(siminf[4]))
                mean_excute_time.append(np.mean(siminf[5]))
                mean_create_time.append(create_time)
                sum_synchronize_time.append(np.sum(siminf[4]))
                sum_excute_time.append(np.sum(siminf[5]))
                sum_create_time.append(create_time*num_thread)
                thread_num.append(num_thread)
                fintime.append(fintime1)
                event_num.append(siminf[6][-1])
                mean_queue.append(np.mean(siminf[1]))
            if fintime1 < min_time:
                optimal_thread = num_thread
                min_time = fintime1

        siminframe = pd.DataFrame(
            {'type': type,'complex': complex,'thread_num': thread_num, 'entity_num': entity_num,'mean_queue':mean_queue,'sum_event':event_num,'mean_create_time': mean_create_time, 'mean_excute2_time': mean_excute2_time,\
             'mean_synchronize_time': mean_synchronize_time, 'mean_excute_time': mean_excute_time,'sum_synchronize_time': sum_synchronize_time, 'sum_excute_time': sum_excute_time,'sum_create_time': sum_create_time,'fintime': fintime, 'optimal thread': optimal_thread})

        if not os.path.exists(output_path):
            wb = Workbook()
            wb.save(output_path)
            with pd.ExcelWriter(output_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                siminframe.to_excel(writer, sheet_name='siminf', index=False)
        else:
            df = pd.read_excel(output_path, engine='openpyxl', sheet_name='siminf')
            row = df.shape[0]
            siminframe = pd.concat([pd.DataFrame(columns=df.columns), siminframe], ignore_index=True)
            book = load_workbook(output_path)
            for sheet in book.sheetnames:
                ws = book[sheet]
                ws.sheet_state = 'visible'  # 将所有工作表设置为可见
            with pd.ExcelWriter(output_path, mode='a',engine='openpyxl', if_sheet_exists='overlay') as writer:
                # writer.book = book  #复制所有原内容
                # writer.sheets = {sheet.title: sheet for sheet in book.worksheets}
                # 追加新数据,追加前必须先格式化新数据,否则新数据缺少列,或是列顺序不对会导致数据紊乱
                siminframe.to_excel(writer, sheet_name='siminf', startrow=row + 1, index=False, header=False)

def newversion_thread_siminf(file_folder,output_path,complex,type):
    files = os.listdir(file_folder)  # 此文件夹下有几个场景  (xxxxxxxx\step0.1-0.8_time2000\data3000_0.2)
    files.sort()
    Entity = {'scenario2':206,'scenario3':400,'scenario4':800,'scenario5':1261,'scenario6':1426}
    for file in files:
        # print(file)
        thread_num = []
        fintime = []
        event_num = []
        entity_num = Entity[file]
        file_path = os.path.join(file_folder,file)  # (xxxxxxxx\step0.1-0.8_time2000\data3000_0.2\scenario2)
        files1 = os.listdir(file_path)  # scenario文件夹下
        files1.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))
        flag = 0
        start_thread = files1[0].split(".")[0].split("_")[2]
        min_time = 1000000
        fintime1 = 1000000
        for filetxt in files1:
            num_thread = int(filetxt.split(".")[0].split("_")[2])  # 工作线程数
            if filetxt[0] == "s":
                flag += 1
                # print(filetxt)
                sim_path = os.path.join(file_path, filetxt)
                print(sim_path)
                try:
                    siminf,fintime1 = read_newversion_siminf(sim_path)
                except:
                    continue

                thread_num.append(num_thread)
                fintime.append(fintime1)
                # event_num.append(siminf[4][-1])
            if fintime1 < min_time:
                optimal_thread = num_thread
                min_time = fintime1

        siminframe = pd.DataFrame(
            {'type':type, 'thread_num': thread_num, 'complex': complex, 'entity_num': entity_num, 'event_num':0,'fintime': fintime, 'optimal thread': optimal_thread})

        if not os.path.exists(output_path):
            wb = Workbook()
            wb.save(output_path)
            with pd.ExcelWriter(output_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                siminframe.to_excel(writer, sheet_name='siminf', index=False)
        else:
            df = pd.read_excel(output_path, engine='openpyxl', sheet_name='siminf')
            row = df.shape[0]
            siminframe = pd.concat([pd.DataFrame(columns=df.columns), siminframe], ignore_index=True)
            book = load_workbook(output_path)
            for sheet in book.sheetnames:
                ws = book[sheet]
                ws.sheet_state = 'visible'  # 将所有工作表设置为可见
            with pd.ExcelWriter(output_path, mode='a',engine='openpyxl', if_sheet_exists='overlay') as writer:
                # writer.book = book  #复制所有原内容
                # writer.sheets = {sheet.title: sheet for sheet in book.worksheets}
                # 追加新数据,追加前必须先格式化新数据,否则新数据缺少列,或是列顺序不对会导致数据紊乱
                siminframe.to_excel(writer, sheet_name='siminf', startrow=row + 1, index=False, header=False)

def pcs_siminf(file_folder1,output_path):
    complexs = os.listdir(file_folder1)  # 此文件夹下有几个复杂度 （xxxxxxxx\X86_PCS\PCS）
    complexs.sort()
    for complex in complexs:
        file_folder = os.path.join(file_folder1, complex)
        cpmlx = complex.split("_")[-1]
        files = os.listdir(file_folder)  # 此文件夹下有几个实体  (xxxxxxxx\AMD_PCS\PCS\v1\complex_0)
        files.sort()
        for file in files:
            thread_num = []
            fintime = []
            # mean_excute1_time = []
            # mean_excute2_time = []
            mean_synchronize_time = []
            sum_synchronize_time = []
            mean_excute_time = []
            sum_excute_time = []
            mean_create_time = []
            sum_create_time = []
            event_num = []
            entity_num = int(file.split("_")[1])
            file_path = os.path.join(file_folder,file)  # (xxxxxxxx\X86_PCS\PCS\complex_0\lp_900_tm_1000)
            files1 = os.listdir(file_path)  # scenario文件夹下
            files1.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))
            flag = 0
            start_thread = files1[0].split(".")[0].split("_")[2]

            min_time = 1000000
            fintime1 = 1000000
            for filetxt in files1:
                num_thread = int(filetxt.split(".")[0].split("_")[2])  # 工作线程数
                if filetxt[0] == "s":
                    flag += 1
                    sim_path = os.path.join(file_path, filetxt)
                    print(sim_path)
                    try:
                        siminf, excute1_time, excute2_time, create_time, fintime1 = read_siminf(num_thread, sim_path)
                    except:
                        continue

                    # mean_excute1_time.append(np.mean(excute1_time))
                    # mean_excute2_time.append(np.mean(excute2_time))
                    mean_synchronize_time.append(np.mean(siminf[4]))
                    mean_excute_time.append(np.mean(siminf[5]))
                    mean_create_time.append(create_time)
                    sum_synchronize_time.append(np.sum(siminf[4]))
                    sum_excute_time.append(np.sum(siminf[5]))
                    sum_create_time.append(create_time*num_thread)
                    thread_num.append(num_thread)
                    fintime.append(fintime1)
                    event_num.append(siminf[6][-1])
                if fintime1 < min_time:
                    optimal_thread = num_thread
                    min_time = fintime1

            # siminframe = pd.DataFrame(
            #     {'thread_num': thread_num, 'entity_num': entity_num, 'event_num': event_num, 'mean_create_time': mean_create_time, 'mean_excute1_time': mean_excute1_time, \
            #      'mean_excute2_time': mean_excute2_time, 'mean_synchronize_time': mean_synchronize_time, 'mean_excute_time': mean_excute_time, 'fintime': fintime, 'optimal thread': optimal_thread})

            siminframe = pd.DataFrame(
                {'complex':cpmlx, 'entity_num': entity_num, 'thread_num': thread_num, 'event_num': event_num, 'mean_create_time': mean_create_time, \
                 'mean_synchronize_time': mean_synchronize_time, 'mean_excute_time': mean_excute_time, \
                 'sum_synchronize_time':sum_synchronize_time,'sum_excute_time':sum_excute_time,'sum_create_time':sum_create_time,'fintime': fintime, 'optimal thread': optimal_thread})

            if not os.path.exists(output_path):
                wb = Workbook()
                wb.save(output_path)
                with pd.ExcelWriter(output_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                    siminframe.to_excel(writer, sheet_name='siminf', index=False)
            else:
                df = pd.read_excel(output_path, engine='openpyxl', sheet_name='siminf')
                row = df.shape[0]
                siminframe = pd.concat([pd.DataFrame(columns=df.columns), siminframe], ignore_index=True)
                book = load_workbook(output_path)
                for sheet in book.sheetnames:
                    ws = book[sheet]
                    ws.sheet_state = 'visible'  # 将所有工作表设置为可见
                with pd.ExcelWriter(output_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    # writer.book = book  #复制所有原内容
                    # writer.sheets = {sheet.title: sheet for sheet in book.worksheets}
                    # 追加新数据,追加前必须先格式化新数据,否则新数据缺少列,或是列顺序不对会导致数据紊乱
                    siminframe.to_excel(writer, sheet_name='siminf', startrow=row + 1, index=False, header=False)




file_folder = 'xxxx\PCS'
output_path = "xxxxx\PCS.xlsx"
files = os.listdir(file_folder)  # 此文件夹下有几个场景  (xxxxxxxx\step0.1-0.8_time2000\data3000_0.2)
files.sort()
for file in files:
    file_path = os.path.join(file_folder,file) #xxxxxxxx\Airline\没有通信交互_intel\v3\nohot_data_1000
    complex = int(file.split("_")[-1])
    # type = file.split("_")[0]
    type = 'exp'
    thread_siminf(file_path,output_path,complex,type)