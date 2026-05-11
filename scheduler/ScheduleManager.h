/*
 * ScheduleManager.h
 *
 * AutoThread (IJCAI/ECAI 2026):
 *   PINO-driven adaptive worker-thread tuning for DES simulators.
 *
 * This header is a slimmed-down reference of the production scheduler
 * used to integrate the PINO predictor (see python_algorithm/PINN.py)
 * with the simulation main loop. Only fields and methods consumed by
 * the AutoThread (dPINN) online-prediction / dynamic-tuning path are
 * retained here; all simulator-internal APIs not related to AutoThread
 * have been removed.
 */

#ifndef GRAPHSIM_SCHEDULEMANAGER_H_
#define GRAPHSIM_SCHEDULEMANAGER_H_

#include <iostream>
#include <fstream>
#include <atomic>
#include <chrono>
#include <map>
#include <tuple>

#include "Schedule.h"
#include "JsonConfiger.h"
#include "workspace/workspace.h"

namespace graphengine {

class ScheduleManager {
public:
    /**
     * Hardware-feature vector consumed by the PINO predictor.
     * The four fields correspond to the four hardware inputs
     * (usr_cpu, system_cpu, VSZ, cswch) defined in the AutoThread paper.
     */
    struct PidstatMetrics {
        double user_cpu   = 0.0;   // %usr
        double system_cpu = 0.0;   // %system
        long   VSZ        = 0;     // virtual memory size (KB)
        double cswch      = 0.0;   // voluntary context switches per second
    };

private:
    bool                 go;              // simulation running flag
    Schedule             schedule_;
    double               localNextTick;
    static ScheduleManager* m_instance;

    std::atomic<size_t>  event_number;

    // Worker-thread pool whose size is tuned online by the PINO predictor
    wsp::workbranch*     branch;

public:
    ScheduleManager();
    virtual ~ScheduleManager();

    /** Singleton accessor. */
    static ScheduleManager* getInstance();

    /**
     * Main simulation loop. Inside this loop the PINO predictor is
     * invoked at workload-shift checkpoints and the load-aware
     * online fine-tuner adjusts `branch` accordingly.
     *
     * @param threads initial worker-thread count
     * @param end     simulation end-time
     */
    void runSimulation(int threads, double end);

    /** Cooperative stop / status helpers used by the tuning logic. */
    void stop();
    void setGo(bool value);
    bool getGo();

    /** Schedule an event into the underlying queue. */
    void scheduleEvent(SimEvent* event_ptr);

    /** Current simulation tick (drives the PINO checkpoint cadence). */
    long double currentTick() { return schedule_.getCurrentTick(); }

    // ------------------------------------------------------------------
    //  PINO feature-collection helpers
    // ------------------------------------------------------------------

    /** Run a shell command and return stdout. Used by `get_pidstat_metrics`. */
    std::string exec(const char* cmd);

    /**
     * Sample the hardware-feature vector for the given pid via `pidstat`.
     * One-shot sampling; slower but does not depend on a background file.
     */
    PidstatMetrics get_pidstat_metrics(pid_t pid, float interval, int count);

    /** Lightweight CPU-usage probe based on /proc/<pid>/stat. */
    std::tuple<double, long, long> get_cpu_usage(pid_t pid);

    /**
     * Read the most recent block of metrics for `pid` from the
     * pidstat output file produced by the trace collector.
     * This is the path used by the AutoThread online predictor.
     */
    PidstatMetrics getLatestStats(pid_t pid);
};

} /* namespace graphengine */

#endif /* GRAPHSIM_SCHEDULEMANAGER_H_ */
