from create_data import *

def writer_timeinf(file_path,num_thread,output_path,scenario,flag=False,sheet_name='sheet1'):

    dataset = create_data(file_path, num_thread)
    train_input = dataset['train_input'].T
    train_output = dataset['train_label'].T
    # [[event_num],[simtime],[walltime],[syntime],[exetime],[exevent]]
    # print(train_input.shape, train_output.shape)

    entity_nums = {'scenario2':206,'scenario3':400,'scenario4':800,'scenario5':1261,'scenario6':1426}
    threads = torch.ones(len(train_input[0,:]))*num_thread
    entity_num = torch.ones(len(train_input[0,:]))*entity_nums[scenario]
    # [[sum_usr],[std_usr],[maxmin_usr],[sum_system],[std_system],[maxmin_system],[sum_wait],[sum_cpu],[std_cpu],[maxmin_cpu],
    # [VSZ],[RSS],[mean_cswch],[std_cswch],maxmin_cswch],[mean_nvcswch],[std_nvcswch],[maxmin_nvcswch],[taskque]],type=tensor

    all_inf = pd.DataFrame({'thread':threads,'sum_usr':train_input[0,:], 'std_usr':train_input[1,:],'maxmin_usr':train_input[2,:],'sum_system':train_input[3,:],'std_system':train_input[4,:],\
                            'maxmin_system':train_input[5,:],'sum_wait':train_input[6,:],'sum_cpu':train_input[7,:],'std_cpu':train_input[8,:],'maxmin_cpu':train_input[9,:],'VSZ':train_input[10,:],\
                            'RSS':train_input[11,:],'mean_cswch':train_input[12,:],'std_cswch':train_input[13,:],'maxmin_cswch':train_input[14,:],'mean_nvcswch':train_input[15,:],\
                            'std_nvcswch':train_input[16,:],'maxmin_nvcswch':train_input[17,:],'taskque':train_input[18,:],'entity_num': entity_num,'event_num':train_output[0,:],\
                            'syntime':train_output[3,:],'exetime':train_output[4,:],'exevent':train_output[5,:],'simtime':train_output[1,:],'walltime':train_output[2,:]})

    if flag:
        all_inf,output_path = writer_csv(all_inf,output_path,sheet_name,mode='a')

    return all_inf,output_path,flag

def writer_allinf(file_path,num_thread,output_path,lp, round, mothed, complex,flag=False,sheet_name='sheet1'):
    dataset, fintime= create_data(file_path, num_thread)
    train_input = dataset['input_overall']
    train_output = dataset['output_overall']
    train_input = train_input.numpy().tolist()
    train_output = train_output.numpy().tolist()
    # [[event_num],[simtime],[walltime],[syntime],[exetime],[exevent]]
    # print(train_input.shape, train_output.shape)

    entity_nums = {'scenario2': 206, 'scenario3': 400, 'scenario4': 800, 'scenario5': 1261, 'scenario6': 1426}
    entity_num = lp
    # [[sum_usr],[std_usr],[maxmin_usr],[sum_system],[std_system],[maxmin_system],[sum_wait],[sum_cpu],[std_cpu],[maxmin_cpu],
    # [VSZ],[RSS],[mean_cswch],[std_cswch],maxmin_cswch],[mean_nvcswch],[std_nvcswch],[maxmin_nvcswch],[taskque]],type=tensor
    # print('sum_usr',train_input[0]),print('std_usr',train_input[1])
    all_inf = pd.DataFrame(
        {'thread': num_thread,'round':round, 'mothed':mothed, 'complex':complex, 'sum_usr': train_input[0], 'std_usr': train_input[1], 'maxmin_usr': train_input[2],
         'sum_system': train_input[3], 'std_system': train_input[4], \
         'maxmin_system': train_input[5], 'sum_wait': train_input[6], 'sum_cpu': train_input[7],
         'std_cpu': train_input[8], 'maxmin_cpu': train_input[9], 'VSZ': train_input[10], \
         'RSS': train_input[11], 'mean_cswch': train_input[12], 'std_cswch': train_input[13],
         'maxmin_cswch': train_input[14], 'mean_nvcswch': train_input[15], \
         'std_nvcswch': train_input[16], 'maxmin_nvcswch': train_input[17], 'taskque': train_output[1],
         'entity_num': entity_num, 'event_num': train_output[0], \
         'syntime': train_output[4], 'exetime': train_output[5], 'exevent': train_output[6],
         'simtime': train_output[2], 'walltime': train_output[3], 'fintime':fintime},index=[0])

    if flag:
        all_inf, output_path = writer_csv(all_inf, output_path, sheet_name, mode='a')

    return all_inf, output_path, flag


##---------------------------------------记录不同线程池调整方法的整体信息------------------------------------------------------
file_folder = "xxxxxxxx\AMD_PCS\Varlog"
output_path = "E:\paper\论文\论文2-机器学习线程与预测\dataset\dynamic_adjust\PCS_AMD.xlsx"
files = os.listdir(file_folder)  # xxxxxxxx\Dynamic
files.sort(reverse=True)
entity_nums = {'scenario1': 102,'scenario2': 206, 'scenario3': 400, 'scenario4': 800, 'scenario5': 1261, 'scenario6': 1426}
for file in files:
    try:
        round = file.split("_")[-1]
    except:
        pass
    file_path1 = os.path.join(file_folder,file) #xxxxxxxx\Dynamic\round_5\
    files1 = os.listdir(file_path1)
    for filetxt in files1:
        mothed = filetxt.split("_")[0]
        complex = filetxt.split("_")[2]
        # thousld = filetxt.split("_")[3] #--新增--
        file_path = os.path.join(file_path1, filetxt) #xxxxxxxx\Dynamic\round_5\ADA_complex_1000
        files2 = os.listdir(file_path)
        for filetxt2 in files2:
            # lp = entity_nums[filetxt2.split("_")[0]]
            lp = filetxt2.split("_")[1]
            file_path2 = os.path.join(file_path, filetxt2) #xxxxxxxx\Dynamic\round_5\ADA_complex_1000\scenario2_staticEntiy
            files3 = os.listdir(file_path2)
            for filetxt3 in files3:
                if filetxt3[0] == 's':
                    thread_num = int(filetxt3.split(".")[0].split("_")[2])
                    writer_allinf(file_path2, thread_num, output_path, lp, round, mothed, complex, True, 'Sheet')


