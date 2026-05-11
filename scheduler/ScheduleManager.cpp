/*
 * ScheduleManager.cpp
 *
 * AutoThread (IJCAI/ECAI 2026):
 *   PINO-driven adaptive worker-thread tuning for DES simulators.
 */

#include "ScheduleManager.h"

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"

#include <unistd.h>
#include <array>
#include <memory>
#include <sstream>
#include <vector>
#include <cmath>

namespace py = pybind11;
extern int event_duration;
extern int C1,C2;
extern std::string OUTPUT_FILE;
extern float thousld;

static std::map<pid_t, std::tuple<std::ifstream, double, long, long, std::chrono::steady_clock::time_point>> pid_streams;
namespace graphengine
{

	ScheduleManager *ScheduleManager::m_instance = new ScheduleManager;

	ScheduleManager *ScheduleManager::getInstance()
	{
		return m_instance;
	}

	ScheduleManager::ScheduleManager()
	{
		localNextTick = 0;
		go = true;
		suspend = false;

		event_number.store(0);

		// 仿真推演开始墙钟时间
		start_walltime = chrono::steady_clock::now();

		// 初始设置仿真为实时运行
		if (JsonConfiger::getInstance()->getConfiger()->speedUp == 0)
		{
			setRealTimeRun(false);
		}
		else
		{
			setRealTimeRun(true);
		}

		// setSpeedRatio(1);

		// 获取本次试验运行的信息
		// configer = JsonConfiger::getInstance()->getConfiger();
	}

	ScheduleManager::~ScheduleManager()
	{
		// 删除线程池、控制线程
		delete m_instance;

		// delete controlthread;
	}

	void ScheduleManager::nextTick()
	{
		localNextTick = schedule_.getNextTick();
		delt_simtime = localNextTick - base_simtime;
	}

	void ScheduleManager::scheduleEvent(SimEvent *event_ptr)
	{
		schedule_.schedule_event(event_ptr);
	}

	void ScheduleManager::setRealTimeRun(bool mode)
	{
		realtime = mode;

		if (realtime)
		{
			resetBaseTime();
		}
	}

	void ScheduleManager::setSpeedRatio(int ratio)
	{
		speedup_ratio = ratio;
	}

	void ScheduleManager::resetBaseTime()
	{
		// 初始化基准时间
		base_simtime = localNextTick;
		base_walltime = chrono::steady_clock::now();

		// 初始化delt时间
		delt_simtime = 0;
	}

	void ScheduleManager::stop()
	{
		go = false;
		cout << "simulaton finished!" << endl;
	}

	bool ScheduleManager::getGo()
	{
		return go;
	}

	void ScheduleManager::setGo(bool value)
	{
		go = value;
	}

	void ScheduleManager::setSuspend(bool value)
	{
		suspend = value;
	}

	void ScheduleManager::runSimulation(int threads, double endTime)
	{
		/*****************启动事件调度线程**************************/
		auto start_create = chrono::steady_clock::now(); // 时间记录
		branch = new wsp::workbranch(threads);
		auto end_create = chrono::steady_clock::now();													   // 时间记录
		auto create_time = chrono::duration_cast<chrono::microseconds>(end_create - start_create).count(); // 时间记录
		SIMLOG_INFO("create {} thread time={} vs", threads, create_time);

		// 初始化localNextTick为当前事件队列中最初始时间的时间戳
		localNextTick = schedule_.getNextTick();

		// AutoThread tuner state (paper Algorithm 1):
		//   flag2 == 1 -> a PINO prediction just happened; verify at C1 cadence
		//   flag  >= 1 -> we are in Level-3 throughput-driven fine-tuning
		int  flag  = 0;
		int  flag2 = 0;
		auto synchronize_walltime = 0L;

		// --------------------pybind文件配置-start-------------------------------------------------

		// **PINN***
		py::scoped_interpreter guard{}; // 启动Python解释器
		py::module sys = py::module::import("sys");
		sys.attr("path").attr("append")("../python_algorithm"); // 添加当前路径到Python模块搜索路径
		py::module_ pinn_module = py::module_::import("PINN");	// 导入自定义Python模块文件
		// in PINO/PINN_mutilhead.ipynb.
		std::string model_path  = R"(../ML_model/model_mutilhead_1.9.pth)";
		std::string scaler_path = R"(../ML_model/feature_scaler.pkl)";
		pinn_module.attr("initialize_model")(model_path, scaler_path); // 初始化函数，只加载一次模型
		py::function pinn_predict = pinn_module.attr("pinn_predict");  // 导入定义的函数
		int last_event_duration = 0;
		double last_eventime = 0.0;
		// --------------------pybind文件配置-end---------------------------------------------------

		// 设置初始加速比
		// setSpeedRatio(JsonConfiger::getInstance()->getConfiger()->speedUp);

		// 离散事件时间推进器
		while ((localNextTick <= endTime) && go)
		{

			// 如果当前是暂停状态，则休息10微秒后
			//  while(suspend)
			//  {
			//  	std::this_thread::sleep_for(std::chrono::microseconds(1));
			//  }

			// 多线程池：从线程池取出多个线程调度执行当前所有时戳相同的事件
			auto start_excute = chrono::steady_clock::now(); // 时间记录 ---------Otter时不需要------------
			printf("当前线程数=%d\n", branch->num_workers());
			// last_event_duration = event_duration;
			schedule_.execute(branch);
			auto end_excute = chrono::steady_clock::now();														   // 时间记录
			auto excute_walltime = chrono::duration_cast<chrono::microseconds>(end_excute - start_excute).count(); // 时间记录
			double current_exetime = excute_walltime * 1.0 / schedule_.return_Events();

			pid_t local_pid = getpid();
			const float load_shift =
				std::fabs(static_cast<float>(last_event_duration - event_duration)) /
				(last_event_duration + 0.1f);

			// ---------- Level 1: load-aware fast prediction ----------
			if (load_shift > thousld && flag == 0 && (int)localNextTick > 30)
			{
				flag2 = 1;
				int thread = branch->num_workers();
				PidstatMetrics metrics = getLatestStats(local_pid);
				long VSZ = 73740L * thread + 246500L;

				py::list hardware_params;
				hardware_params.append(metrics.user_cpu);
				hardware_params.append(metrics.system_cpu);
				hardware_params.append(VSZ);
				hardware_params.append(metrics.cswch);

				float mu_e = 1.0f / event_duration;
				int   N    = schedule_.return_Events();
				py::object result = pinn_predict(N, mu_e, thread, hardware_params);
				int prediction = std::min(result.cast<int>(), 60);

				while (branch->num_workers() < prediction) branch->add_worker();
				while (branch->num_workers() > prediction) branch->del_worker();
				last_eventime = current_exetime;
			}
			// ---------- Level 2: CPU-aware periodic adjustment ----------
			// Short cadence C1 right after a PINO update; long cadence C2 otherwise.
			else if ((flag2 == 1 && (int)localNextTick % C1 == 0) ||
					 (flag2 == 0 && (int)localNextTick % (C1 * C2) == 0))
			{
				PidstatMetrics metrics = getLatestStats(local_pid);
				if (metrics.user_cpu < metrics.system_cpu && branch->num_workers() > 3)
				{
					branch->del_worker();
					flag = 1;            // enter Level-3 fine-tuning next cycle
				}
				last_eventime = current_exetime;
				flag2 = 0;
			}
			// ---------- Level 3: throughput-driven fine-tuning ----------
			// Lower per-event time means higher throughput; only ever *remove*
			// threads (paper: undersubscription < oversubscription cost).
			else if (flag >= 1 && branch->num_workers() > 3)
			{
				if (current_exetime < last_eventime)
				{
					branch->del_worker();
					last_eventime = current_exetime;
					flag = 1;
				}
				else if (flag < 3)
				{
					++flag;              // tolerate up to 3 noisy cycles
				}
				else
				{
					branch->add_worker();   // rollback last reduction
					last_eventime = current_exetime;
					flag = 0;            // exit Level 3
				}
			}
			
			// 同步处理过SimElement状态信息
			auto start_synchronize = chrono::steady_clock::now(); // 时间记录
			schedule_.synchronize();

			// 推进仿真时间
			localNextTick = schedule_.getNextTick();

			// 计算设置实时仿真后的累计墙钟时间
			auto now_walltime = chrono::steady_clock::now();
			// auto delt_walltime = chrono::duration_cast<chrono::milliseconds>(now_walltime - base_walltime).count();

			
			auto end_synchronize = chrono::steady_clock::now();																 // 时间记录
			synchronize_walltime = chrono::duration_cast<chrono::microseconds>(end_synchronize - start_synchronize).count(); // 时间记录

			// 仿真推演到当前总的墙钟时间
			auto total_walltime = chrono::duration_cast<chrono::milliseconds>(now_walltime - start_walltime).count();

			// 控制台输出仿真时钟推进信息
			//  SIMLOG_INFO("simulation time={}, wall time={} ms",localNextTick, total_walltime);
			//  SIMLOG_INFO("simulation time={}, wall time={} ms, executed events={}",localNextTick, total_walltime, schedule_.getEventCount());
			SIMLOG_INFO("simulation time={}, wall time={} ms,  excute time={} vs,  synchronize time={} vs, executed events={}", localNextTick, total_walltime, excute_walltime, synchronize_walltime, schedule_.getEventCount());

			// Algorithm 1, line 18: t_prev <- t_cur for the next-cycle load-shift test
			last_event_duration = event_duration;
		}
	}

	std::string ScheduleManager::exec(const char *cmd)
	{
		std::array<char, 128> buffer;
		std::string result;
		std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);

		if (!pipe)
		{
			throw std::runtime_error("popen() failed!");
		}

		while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
		{
			result += buffer.data();
		}

		return result;
	}

	ScheduleManager::PidstatMetrics ScheduleManager::get_pidstat_metrics(pid_t pid, float interval, int count)
	{
		PidstatMetrics metrics;
		// 构建命令
		std::ostringstream cmd_stream;
		cmd_stream << "pidstat " << "-urdwh" << " -p " << pid << " " << interval << " " << count;
		std::string cmd = cmd_stream.str();

		// 执行命令
		std::string output;
		try
		{
			output = exec(cmd.c_str());
		}
		catch (const std::runtime_error &e)
		{
			std::cerr << "Error executing pidstat: " << e.what() << std::endl;
			return metrics;
		}

		// 解析输出
		// 找到包含CPU利用率的数据行（最后一行）
		std::istringstream output_stream(output);
		std::string line;
		std::vector<std::string> lines;

		while (std::getline(output_stream, line))
		{
			if (!line.empty())
			{
				lines.push_back(line);
			}
		}

		// 需要至少3行输出（标题行+数据行）
		if (lines.size() < 3)
		{
			std::cerr << "Unexpected pidstat output format" << std::endl;
			return metrics;
		}

		// 获取最后一行（最新数据）
		std::string data_line = lines[lines.size() - 1];
		std::istringstream data_stream(data_line);
		std::vector<std::string> tokens;
		std::string token;

		// 令牌化数据行
		while (data_stream >> token)
		{
			tokens.push_back(token);
		}

		// 预期的token数量
		// 典型格式：时间 UID PID %usr %system %guest %wait %CPU CPU Command
		if (tokens.size() < 3)
		{
			std::cerr << "Unexpected token count in pidstat output: " << tokens.size() << std::endl;
			return metrics;
		}

		// 尝试提取CPU利用率值
		try
		{
			metrics.user_cpu = std::stod(tokens[3]);
			metrics.system_cpu = std::stod(tokens[4]);
			metrics.VSZ = std::stod(tokens[11]);
			metrics.cswch = std::stod(tokens[18]);
		}
		catch (const std::exception &e)
		{
			std::cerr << "Error parsing pidstat values: " << e.what() << std::endl;
			std::cerr << "Problematic token: " << tokens[3] << ", " << tokens[7] << ", " << tokens[11] << std::endl;
		}
		return metrics;
	}

	tuple<double, long, long> ScheduleManager::get_cpu_usage(pid_t pid)
	{
		auto &[stat_file, last_cpu_usage, prev_utime, prev_stime, prev_time] = pid_streams[pid];
		// 如果是首次调用该PID，初始化文件流和状态
		if (!stat_file.is_open())
		{
			std::string stat_path = "/proc/" + std::to_string(pid) + "/stat";
			stat_file.open(stat_path);
			// 文件打开失败的处理
			if (!stat_file)
			{
				pid_streams.erase(pid); // 清理无效条目
				return {0.0, 0, 0};		// 返回错误值
			}
			// 初始状态设置
			last_cpu_usage = 0.0;
			prev_utime = prev_stime = 0;
			prev_time = std::chrono::steady_clock::now();
		}
		// 重置到文件起始位置读取最新数据
		stat_file.clear();
		stat_file.seekg(0);
		std::string line;
		// 读取失败的处理（进程可能已结束）
		if (!std::getline(stat_file, line))
		{
			stat_file.close();
			pid_streams.erase(pid);
			return {0.0, 0, 0};
		}
		// 解析stat文件
		std::istringstream ss(line); // 将读取的行拆分成多个字段（空格分隔）
		std::string token;
		long utime = 0, stime = 0; // 存储用户态和内核态CPU时间
		// 跳过前13个字段
		for (int i = 0; i < 13; ++i)
		{
			if (!(ss >> token))
			{ // 确保不会在意外结束时崩溃// 跳过前13个字段（如进程名、状态、父进程ID等）。
				return {last_cpu_usage, prev_utime, prev_stime};
			}
		}
		// 读取utime(14)和stime(15)
		ss >> utime >> stime;
		auto now = std::chrono::steady_clock::now();
		double time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - prev_time).count() / 1000.0;
		prev_time = now;
		// 计算CPU时间差
		long total_time_diff = (utime + stime) - (prev_utime + prev_stime);
		prev_utime = utime;
		prev_stime = stime;
		if (time_elapsed <= 0)
			return {last_cpu_usage, utime, stime}; // 避免除以零
		// 计算CPU使用率 (总CPU时间/实际时间)
		double cpu_usage = (total_time_diff / sysconf(_SC_CLK_TCK)) / time_elapsed;
		// 平滑变化
		last_cpu_usage = 0.8 * last_cpu_usage + 0.2 * cpu_usage;
		// 返回 (cpu_usage, utime, stime)
		return {last_cpu_usage, utime, stime};
	}

	// 从文件获取最新统计数据
	ScheduleManager::PidstatMetrics ScheduleManager:: getLatestStats(pid_t pid)
	{
		PidstatMetrics stats = {580.2739683, 21.71698413, 684540, 1972.190762};
		std::ifstream file(OUTPUT_FILE);
		if (!file.is_open())
		{
			std::cerr << "Error opening output file: " << OUTPUT_FILE << std::endl;
			return stats;
		}

		std::string pid_str = std::to_string(pid); //将传入的pid（整数）转换为字符串形式，用于后续匹配
		// std::string pid_str = "11591";
		std::vector<std::string> relevant_lines; //创建一个字符串向量，用于存储与目标pid相关的行（最后一段时间块中匹配到的行）

		// 存储最后一段时间段的数据块
		std::vector<std::string> last_block; //用于存储当前正在读取的最后一段时间块的数据行（针对当前pid的行）
		std::string line;

		while (std::getline(file, line)) //循环读取文件的每一行
		{
			// 忽略空行和注释行
			if (line.empty() || line[0] == '#' || line[0] == 'L' || (line.find("|__") != std::string::npos))
				continue;

			// 识别时间戳行（新数据块的开始）
			if (line.find("时") != std::string::npos) //检查该行是否包含中文字符“时”
			{
				if (!last_block.empty())
				{
					relevant_lines = last_block; // 前一个完整数据块
					last_block.clear();
				}
			}

			// 检查PID匹配
			if (line.find(pid_str) != std::string::npos)
			{
				last_block.push_back(line);
			}
		}
		// 处理找到的相关行（只取第一个TID为'-'的行）
		bool found_main = false; //用于记录是否找到了代表整个进程的数据行（TID为'-'）
		for (const auto &data_line : last_block) //遍历relevant_lines中的每一行
		{
			// 按空格分割行
			std::istringstream iss(data_line); //使用字符串流iss来分割当前行
			std::vector<std::string> tokens; //存储分割后的单词
			std::string token;

			while (iss >> token)
			{
				tokens.push_back(token); //将当前行按空格分割，存入tokens
			}

			// 检查TID是否为'-'（表示整个进程的数据）
			if (tokens.size() >= 20 && tokens[3] == "-")
			{
				try
				{
					// 提取所需的数据
					stats.user_cpu = std::stof(tokens[4]);	 // %usr
					stats.system_cpu = std::stof(tokens[5]); // %system
					stats.VSZ = std::stol(tokens[12]);	 // VSZ (已经是KB单位)
					stats.cswch = std::stof(tokens[19]); // cswch/s
					found_main = true;
					break; // 只取第一行TID为'-'的数据
				}
				catch (const std::exception &e)
				{
					std::cerr << "Error parsing data line: " << e.what() << std::endl;
				}
			}
		}

		if (!found_main)
		{
			std::cerr << "No valid TGID line found for PID: " << pid << std::endl;
		}

		return stats;
	}
} /* namespace graphsim */
