import os
import pandas as pd
from openpyxl import Workbook,load_workbook

def read_Hotsport(input_path,out_path):
    files = os.listdir(input_path)
    files.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))
    num_thread = []
    Elapsed_Time = []
    CPU_Time = []
    Effective_Time = []
    Spin_Time = []
    Overhead_Time = []
    Instructions_Retired = []
    Mic_Usage = []
    CPI_Rate = []
    name_1 = []; value_1 = []; name_2 = []; value_2 = [];name_3 = []; value_3 = []; name_4 = []; value_4 = [];name_5 = []; value_5 = []
    for file in files:
        if file[0] == "2":
            num_thread.append(int(file.split(".")[0].split("_")[2]))
            path = os.path.join(input_path,file)
            file_new = open(path, "r", encoding="utf-8")
            lines = file_new.readlines()
            Time = []
            for i in range(6):
                new_line = lines[i].split(" ")
                value = new_line[-1][:-2]
                Time.append(value)
            new_line = lines[6].split(" ")
            value = new_line[-4][:-1]
            Mic_Usage.append(float(value)/100)
            # print(Time)
            Elapsed_Time.append(float(Time[0]))
            CPU_Time.append(float(Time[1]))
            Effective_Time.append(float(Time[2]))
            Spin_Time.append(float(Time[3]))
            Overhead_Time.append(float(Time[4]))
            Instructions_Retired.append(Time[5])
            new_line = lines[15].split(" ")
            value = new_line[-1]
            CPI_Rate.append(float(value))
            if lines[22][0] == ' ':
                line_1 = lines[27].split("  ")
                name_1.append(line_1[0])
                value_1.append(float(line_1[-1][:-2])/100)
                line_2 = lines[28].split("  ")
                name_2.append(line_2[0])
                value_2.append(float(line_2[-1][:-2])/100)
                line_3 = lines[29].split("  ")
                name_3.append(line_3[0])
                value_3.append(float(line_3[-1][:-2])/100)
                line_4 = lines[30].split("  ")
                name_4.append(line_4[0])
                value_4.append(float(line_4[-1][:-2])/100)
                line_5 = lines[31].split("  ")
                name_5.append(line_5[0])
                value_5.append(float(line_5[-1][:-2])/100)
            else:
                line_1 = lines[22].split("  ")
                name_1.append(line_1[0])
                value_1.append(float(line_1[-1][:-2])/100)
                line_2 = lines[23].split("  ")
                name_2.append(line_2[0])
                value_2.append(float(line_2[-1][:-2])/100)
                line_3 = lines[24].split("  ")
                name_3.append(line_3[0])
                value_3.append(float(line_3[-1][:-2])/100)
                line_4 = lines[25].split("  ")
                name_4.append(line_4[0])
                value_4.append(float(line_4[-1][:-2])/100)
                line_5 = lines[26].split("  ")
                name_5.append(line_5[0])
                value_5.append(float(line_5[-1][:-2])/100)



    outdata = pd.DataFrame(
        {'Elapsed_Time': Elapsed_Time, 'CPU_Time': CPU_Time, 'Effective_Time': Effective_Time,'Spin_Time': Spin_Time,'Overhead_Time': Overhead_Time, \
         'Instructions_Retired': Instructions_Retired,'Mic_Usage':Mic_Usage,'CPI_Rate':CPI_Rate,'name_1':name_1,'value_1':value_1,'name_2':name_2,
         'value_2':value_2,'name_3':name_3,'value_3':value_3,'name_4':name_4,'value_4':value_4,'name_5':name_5,'value_5':value_5})
    # print(outdata.columns)
    if not os.path.exists(out_path):
        wb = Workbook()
        wb.save(out_path)
        with pd.ExcelWriter(out_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            outdata.to_excel(writer, sheet_name='Sheet1', index=False)
    else:
        df = pd.read_excel(out_path, engine='openpyxl', sheet_name='Sheet1')
        row = df.shape[0]
        out_path = pd.concat([pd.DataFrame(columns=df.columns), out_path], ignore_index=True)
        book = load_workbook(out_path)
        for sheet in book.sheetnames:
            ws = book[sheet]
            ws.sheet_state = 'visible'  # 将所有工作表设置为可见
        with pd.ExcelWriter(out_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # writer.book = book  #复制所有原内容
            # writer.sheets = {sheet.title: sheet for sheet in book.worksheets}
            # 追加新数据,追加前必须先格式化新数据,否则新数据缺少列,或是列顺序不对会导致数据紊乱
            out_path.to_excel(writer, sheet_name='Sheet1', startrow=row + 1, index=False, header=False)


# input_path = "D:\Program Files\Intel\projects\\report_outpath\Autorun_Hotspots_2000"
# out_path = "E:\paper\论文\论文2-机器学习线程与预测\dataset\\vtune_report\Autorun_Hotsport_2000.xlsx"
# read_Hotsport(input_path, out_path)


def read_Threading(input_path,out_path):
    files = os.listdir(input_path)
    files.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))
    num_thread = []
    Elapsed_Time = []
    Inactive_Wait_Time0 = []
    Inactive_Sync_Wait = []
    Preemption_Wait_Time0 = []
    Spin_Overhead_Time = []

    name_1 = [[] for i in range(5)]
    Inactive_Wait_Time = [[] for i in range(5)]
    Inactive_Sync_Time = [[] for i in range(5)]
    Inactive_Sync_Count = [[] for i in range(5)]
    Preemption_Wait_Time = [[] for i in range(5)]
    Preemption_Wait_Count = [[] for i in range(5)]

    name_2 = [[] for i in range(5)]
    Spin_Overhead_Time1 = [[] for i in range(5)]
    From_CPU_Time = [[] for i in range(5)]

    for file in files:
        if file[0] == "4":
            num_thread.append(int(file.split(".")[0].split("_")[2]))
            path = os.path.join(input_path, file)
            file_new = open(path, "r", encoding="utf-8")
            lines = file_new.readlines()
            Time = []
            line_1 = lines[0].split(': ')
            Elapsed_Time.append(float(line_1[1][:-2]))
            for i, line in enumerate(lines, start=1):
                if "Inactive Wait Time with poor CPU Utilization:" in line:
                    line_2 = line.replace(": ", " ");line_2 = line_2.split(' ');
                    Inactive_Wait_Time0.append(float(line_2[-6][:-1]))
                elif "Inactive Sync Wait Time: " in line:
                    line_3 = line.split(': ');
                    Inactive_Sync_Wait.append(float(line_3[-1][:-2]))
                elif "Preemption Wait Time: " in line:
                    line_4 = line.split(': ');
                    Preemption_Wait_Time0.append(float(line_4[-1][:-2]))
                elif "Top functions by Inactive Wait Time" in line:
                    for iter in range(5):
                        line_1 = lines[i+iter+2].split()
                        name_1[iter].append(line_1[0])
                        Inactive_Wait_Time[iter].append(float(line_1[-5][:-1]))
                        Inactive_Sync_Time[iter].append(float(line_1[-4][:-1]))
                        Inactive_Sync_Count[iter].append(line_1[-3])
                        Preemption_Wait_Time[iter].append(float(line_1[-2][:-1]))
                        Preemption_Wait_Count[iter].append(line_1[-1])
                elif "Spin and Overhead Time: " in line:
                    line = line.split(" ")
                    Spin_Overhead_Time.append(float(line[-5][:-1]))
                elif "Top Functions with Spin or Overhead Time" in line:
                    for iter in range(5):
                        line_2 = lines[i+iter+2].split()
                        name_2[iter].append(line_2[0])
                        Spin_Overhead_Time1[iter].append(float(line_2[-2][:-1]))
                        From_CPU_Time[iter].append(float(line_2[-1][:-1])/100.0)
                    break

    outdata = pd.DataFrame(
        {'Elapsed_Time': Elapsed_Time, 'Inactive_Wait_Time_all': Inactive_Wait_Time0, 'Inactive_Sync_Wait_all': Inactive_Sync_Wait, 'Preemption_Wait_Time_all': Preemption_Wait_Time0,\
         'name_11': name_1[0], 'Inactive_Wait_Time1': Inactive_Wait_Time[0], 'Inactive_Sync_Time1': Inactive_Sync_Time[0], 'Inactive_Sync_Count1': Inactive_Sync_Count[0],\
         'Preemption_Wait_Time1': Preemption_Wait_Time[0],'Preemption_Wait_Count1': Preemption_Wait_Count[0], \
         'name_12': name_1[1], 'Inactive_Wait_Time2': Inactive_Wait_Time[1],'Inactive_Sync_Time2': Inactive_Sync_Time[1], 'Inactive_Sync_Count2': Inactive_Sync_Count[1], \
         'Preemption_Wait_Time2': Preemption_Wait_Time[1], 'Preemption_Wait_Count2': Preemption_Wait_Count[1], \
         'name_13': name_1[2], 'Inactive_Wait_Time3': Inactive_Wait_Time[2],'Inactive_Sync_Time3': Inactive_Sync_Time[2], 'Inactive_Sync_Count3': Inactive_Sync_Count[2], \
         'Preemption_Wait_Time3': Preemption_Wait_Time[2], 'Preemption_Wait_Count3': Preemption_Wait_Count[2], \
         'name_14': name_1[3], 'Inactive_Wait_Time4': Inactive_Wait_Time[3],'Inactive_Sync_Time4': Inactive_Sync_Time[3], 'Inactive_Sync_Count4': Inactive_Sync_Count[3], \
         'Preemption_Wait_Time4': Preemption_Wait_Time[3], 'Preemption_Wait_Count4': Preemption_Wait_Count[3], \
         'name_15': name_1[4], 'Inactive_Wait_Time5': Inactive_Wait_Time[4],'Inactive_Sync_Time5': Inactive_Sync_Time[4], 'Inactive_Sync_Count5': Inactive_Sync_Count[4], \
         'Preemption_Wait_Time5': Preemption_Wait_Time[4], 'Preemption_Wait_Count5': Preemption_Wait_Count[4], \
         'Spin_Overhead_Time_all': Spin_Overhead_Time, 'name_21': name_2[0], 'Spin_Overhead_Time1': Spin_Overhead_Time1[0], 'From_CPU_Time1': From_CPU_Time[0], \
         'name_22': name_2[1], 'Spin_Overhead_Time2': Spin_Overhead_Time1[1], 'From_CPU_Time2': From_CPU_Time[1], \
         'name_23': name_2[2], 'Spin_Overhead_Time3': Spin_Overhead_Time1[2], 'From_CPU_Time3': From_CPU_Time[2], \
         'name_24': name_2[3], 'Spin_Overhead_Time4': Spin_Overhead_Time1[3], 'From_CPU_Time4': From_CPU_Time[3], \
         'name_25': name_2[4], 'Spin_Overhead_Time5': Spin_Overhead_Time1[4], 'From_CPU_Time5': From_CPU_Time[4]})
    # print(outdata.columns)
    if not os.path.exists(out_path):
        wb = Workbook()
        wb.save(out_path)
        with pd.ExcelWriter(out_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            outdata.to_excel(writer, sheet_name='Sheet1', index=False)
    else:
        df = pd.read_excel(out_path, engine='openpyxl', sheet_name='Sheet1')
        row = df.shape[0]
        out_path = pd.concat([pd.DataFrame(columns=df.columns), out_path], ignore_index=True)
        book = load_workbook(out_path)
        for sheet in book.sheetnames:
            ws = book[sheet]
            ws.sheet_state = 'visible'  # 将所有工作表设置为可见
        with pd.ExcelWriter(out_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # writer.book = book  #复制所有原内容
            # writer.sheets = {sheet.title: sheet for sheet in book.worksheets}
            # 追加新数据,追加前必须先格式化新数据,否则新数据缺少列,或是列顺序不对会导致数据紊乱
            out_path.to_excel(writer, sheet_name='Sheet1', startrow=row + 1, index=False, header=False)


input_path = "xxxx\GraphSim_Threading_2000"
out_path = "xxxxx\Graphsim_Threading_2000.xlsx"
read_Threading(input_path, out_path)